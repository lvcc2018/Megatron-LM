# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Finetune utilities."""

from functools import partial
import sys
import torch
from typing import List
import math

from megatron import get_args, get_num_microbatches
from megatron import print_rank_0, print_rank_last
from megatron import get_timers
from megatron import is_last_rank
from megatron.core import mpu
from megatron.core.utils import get_model_config
from megatron.core.enums import ModelType
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.training import setup_model_and_optimizer
from megatron.training import train_step
from megatron.training import training_log
from megatron.utils import average_losses_across_data_parallel_group
from megatron.utils import calc_params_l2_norm
from megatron.utils import check_adlr_autoresume_termination
from megatron.model.vision.knn_monitor import compute_feature_bank
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron import get_tensorboard_writer, get_writer
from tasks.instruction.DataLoaderManager import DataLoaderManager, ConstantRateScheduler


def process_batch(batch):
    """Process batch and produce inputs for the model."""
    args = get_args()

    tokens = batch['text'].long().cuda().contiguous()
    types = batch['types'].long().cuda().contiguous()
    labels = batch['label'].long().cuda().contiguous()
    attention_mask = batch['padding_mask'].float().cuda().contiguous()
    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, types, labels, attention_mask


def cross_entropy_loss_func(labels, output_tensor):
    logits = output_tensor

    # Cross-entropy loss.
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(logits.contiguous().float(), labels)

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def _cross_entropy_forward_step(batch, model):
    """Simple forward step with cross-entropy loss."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch
    tokens, types, labels, attention_mask = process_batch(batch_)
    timers('batch-generator').stop()

    # Forward model.
    print_rank_last(batch)
    quit()
    output_tensor = model(tokens, attention_mask, tokentype_ids=types)

    return output_tensor, partial(cross_entropy_loss_func, labels)


def build_data_loader(datasets, micro_batch_size, num_workers, drop_last,
        task_collate_fn=None, shuffle=True, sample=True):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""

    # Sampler.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    def get_dataloader(dataset):
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last)

        # Data loader. Note that batch size is the per GPU batch size.
        data_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=micro_batch_size,
                                                sampler=sampler,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                drop_last=drop_last,
                                                pin_memory=True,
                                                collate_fn=task_collate_fn)
        return data_loader
    # if len(datasets) > 1:
    if isinstance(datasets, list):
        args = get_args()
        data_loaders = [(get_dataloader(d), args.seed, d.dataset_name, ConstantRateScheduler(d.weight)) for d in datasets]
        data_loader = DataLoaderManager(data_loaders, rank=args.rank, world_size=args.world_size, gradient_accumulation_steps=get_num_microbatches(), seed=args.seed, sample=sample)
    else:
        data_loader = get_dataloader(datasets[0])
    
    return data_loader


def _build_infinite_size_dataloader(dataloader):
    """Build a looped dataloader with infinite size."""

    iterator = dataloader.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = dataloader.__iter__()


def _build_train_valid_dataloaders(train_dataset, valid_dataset, 
    task_collate_fn=None):
    """Traing and validation dataloaders."""
    args = get_args()

    print_rank_last('building train and validation dataloaders ...')
    # Training dataset.
    train_dataloader = build_data_loader(train_dataset, args.micro_batch_size,
                                         args.num_workers, not args.keep_last,
                                         task_collate_fn, shuffle=True, sample=True)
    # Set the training iterations.
    if not isinstance(train_dataloader, DataLoaderManager):
        args.train_iters_per_epoch = len(train_dataloader) // get_num_microbatches()
        args.train_iters = args.epochs * args.train_iters_per_epoch
    else:
        assert args.train_iters is not None
        args.train_iters_per_epoch = args.train_iters
        args.epochs = 1
    # Validation dataset. For this dataset, we do not need to set up
    # shuffling so we can just use a simple infinite loop.
    if valid_dataset is not None:
        valid_dataloader = build_data_loader(valid_dataset, args.micro_batch_size,
                                            args.num_workers, not args.keep_last,
                                            task_collate_fn, shuffle=False, sample=False)
        # valid_dataloader = _build_infinite_size_dataloader(valid_dataloader_)
        # Set the training iterations.
        if isinstance(valid_dataloader, DataLoaderManager):
            args.eval_iters = valid_dataloader.one_epoch_iters()
        else:
            args.eval_iters = len(valid_dataloader)
        print_rank_last(f'set evaluation iters to {args.eval_iters}')
    else:
        valid_dataloader = None

    # Now that we've built the data loaders, set batch_size arguments
    # to the actual batch size the model will see for this dataset.
    # This is necessary so pipeline transfers know what size they are
    # and the LR schedule, which is based on samples seen, gets set
    # correctly.
    args.orig_micro_batch_size = args.micro_batch_size
    args.orig_global_batch_size = args.global_batch_size
    if hasattr(train_dataset, 'sample_multiplier'):
        # If our dataset as a sample_multiplier attribute that means
        # each "sample" from the dataset actually has multiple samples
        # that will collapse into the batch dimension (for example in
        # the RACE dataset that has several options), we need to
        # account for that when setting the micro batch size.
        args.micro_batch_size *= train_dataset.sample_multiplier
        args.global_batch_size *= train_dataset.sample_multiplier

    return train_dataloader, valid_dataloader


def _train(model, optimizer, opt_param_scheduler, forward_step,
           train_dataloader, valid_dataloader, end_of_epoch_callback, config):
    """Train the model."""
    args = get_args()
    timers = get_timers()

    # assert get_num_microbatches() == 1, "finetuning with gradient accumulation doesn't currently work"

    # Turn on training mode which enables dropout.
    for m in model:
        m.train()

    # Tracking loss.
    losses_dict_sum = {}
    if isinstance(train_dataloader, DataLoaderManager):
        dataset_iters = {}
    else:
        dataset_iters = None

    # Starting epoch and iteration
    start_epoch = args.iteration // args.train_iters_per_epoch
    start_iteration = args.iteration % args.train_iters_per_epoch
    iteration = args.iteration
    gradient_accumulation_steps = get_num_microbatches()

    # Memory reporting flag.
    report_memory_flag = True
    
    # Setup some training config params
    config.grad_scale_func = optimizer.scale_loss
    config.timers = timers
    
    # For each remaining epoch
    timers('interval-time', log_level=0).start(barrier=True)
    for epoch in range(start_epoch, args.epochs):
        print_rank_last('working on epoch {} ...'.format(epoch + 1))
        if not isinstance(train_dataloader, DataLoaderManager):
            # Set the data loader epoch to shuffle the index iterator.
            train_dataloader.sampler.set_epoch(args.seed + epoch)
        train_iterator = enumerate(train_dataloader)
        for i in range(gradient_accumulation_steps * start_iteration):
            # Ignore the iterations before starting value
            b = next(train_iterator)
        for _ in range(start_iteration, args.train_iters_per_epoch):
            # Train for one step.
            out = train_step(forward_step, train_iterator, model, 
                             optimizer, opt_param_scheduler, config)
        # # For all the batches in the dataset.
        # for iteration_, batch in enumerate(train_dataloader):

        #     # Ignore the iterations before starting value
        #     if iteration_ < start_iteration:
        #         continue
        #     # Set to zero so the next epoch does not skip any batches.
        #     start_iteration = 0

        #     # Train for one step.
        #     out = train_step(forward_step, batch, model, optimizer, opt_param_scheduler)

            losses_dict, skipped_iter, grad_norm, num_zeros_in_grad = out
            iteration += 1

            # Logging.
            params_norm = None
            if args.log_params_norm:
                params_norm = calc_params_l2_norm(model)
            epoch_states = None
            if isinstance(train_dataloader, DataLoaderManager):
                epoch_states = train_dataloader.get_states()
            report_memory_flag = training_log(losses_dict, losses_dict_sum,
                                              optimizer.param_groups[0]['lr'],
                                              iteration,
                                              optimizer.get_loss_scale().item(),
                                              report_memory_flag, skipped_iter,
                                              grad_norm, params_norm, num_zeros_in_grad,
                                              epoch_states, dataset_iters)

            # Autoresume
            if args.adlr_autoresume and \
               (iteration % args.adlr_autoresume_interval == 0):
                check_adlr_autoresume_termination(iteration, model,
                                                  optimizer, opt_param_scheduler)

            # Checkpointing
            saved_checkpoint = False
            if args.save and args.save_interval and \
               iteration % args.save_interval == 0:
                save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
                saved_checkpoint = True

            # Evaluation
            if args.eval_interval and iteration % args.eval_interval == 0:
                prefix = 'iteration {}'.format(iteration)
                _evaluate_and_print_results(prefix, forward_step,
                                           enumerate(valid_dataloader), model,
                                           iteration, None, True, 1)

            # Exiting based on iterations
            if args.exit_interval and iteration % args.exit_interval == 0:
                if not saved_checkpoint:
                    save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
                torch.distributed.barrier()
                print_rank_last('exiting program at iteration {}'.format(iteration))
                sys.exit()

        # Set to zero so the next epoch does not skip any batches.
        start_iteration = 0
        # Checkpointing at the end of each epoch.
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)

        # Callback at the end of each epoch.
        if end_of_epoch_callback is not None:
            end_of_epoch_callback(model, epoch)

def _evaluate(forward_step_func,
             data_iterator,
             model,
             process_non_loss_data_func,
             verbose=False,
             gradient_accumulation_steps=get_num_microbatches()):
    """Evaluation."""
    args = get_args()

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    total_loss_dict = {}
    total_eval_iters = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))

            forward_backward_func = get_forward_backward_func()
            loss_dicts = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=gradient_accumulation_steps,
                dtype=args.params_dtype,
                tensor_shape=(args.seq_length,
                              args.micro_batch_size, args.hidden_size),
                sequence_parallel=args.sequence_parallel,
                forward_only=True,
                timers=None)
                # forward_step_func, data_iterator, model, optimizer=None,
                # timers=None, forward_only=True, gradient_accumulation_steps=gradient_accumulation_steps)
            # print_rank_last(loss_dicts)

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        total_loss_dict[key] = total_loss_dict.get(
                            key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
                        total_eval_iters[key] = total_eval_iters.get(
                            key, torch.cuda.FloatTensor([0.0])) + 1

            args.consumed_valid_samples += mpu.get_data_parallel_world_size() \
                                           * args.micro_batch_size
        collected_non_loss_data = None
        if process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func, data_iterator, model, optimizer=None,
                timers=None, forward_only=True, collect_non_loss_data=True)

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= total_eval_iters[key]

    return total_loss_dict, collected_non_loss_data

def _evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, process_non_loss_data_func,
                               verbose=False,
                               gradient_accumulation_steps=get_num_microbatches()):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    # writer = get_tensorboard_writer()
    writer = get_writer()

    total_loss_dict, collected_non_loss_data = _evaluate(
        forward_step_func, data_iterator, model,
        process_non_loss_data_func, verbose, gradient_accumulation_steps)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.4E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        # string += '{} PPL: {:.4E} | '.format(key, ppl)
        if writer:
            writer.add_scalar('validation/{}'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            # writer.add_scalar('validation/{} vs samples'.format(key),
            #                   total_loss_dict[key].item(),
            #                   args.consumed_train_samples)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('validation ppl/{}'.format(key), ppl,
                                  iteration)
                # writer.add_scalar('validation ppl/{} vs samples'.format(key),
                #                   ppl, args.consumed_train_samples)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)


def finetune(train_valid_datasets_provider, model_provider,
             model_type=ModelType.encoder_or_decoder,
             forward_step=_cross_entropy_forward_step,
             end_of_epoch_callback_provider=None,
             task_collate_fn=None):
    """Main finetune function used across all tasks."""
    args = get_args()
    timers = get_timers()

    assert args.rampup_batch_size is None, \
        'batch size scaling is not supported for finetuning'

    # Train and validation data loaders.
    timers('train/valid/test dataset/dataloder', log_level=0).start()
    if args.epochs > 0:
        train_dataset, valid_dataset = train_valid_datasets_provider()
        train_dataloader, valid_dataloader = _build_train_valid_dataloaders(
            train_dataset, valid_dataset, task_collate_fn)
    else:
        args.train_iters = 0
    timers('train/valid/test dataset/dataloder').stop()

    # Build calback function.
    timers('callback function', log_level=0).start()
    end_of_epoch_callback = None
    if end_of_epoch_callback_provider is not None:
        end_of_epoch_callback = end_of_epoch_callback_provider()
    timers('callback function').stop()

    # Build model, optimizer and learning rate scheduler.
    timers('model and optimizer', log_level=0).start()
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(model_provider, model_type)
    timers('model and optimizer').stop()
    config = get_model_config(model[0])
    
    # If pretrained checkpoint is provided and we have not trained for
    # any iteration (i.e., iteration is zero), then load the pretrained
    # checkpoint.
    timers('pretrained checkpoint', log_level=0).start(barrier=True)
    if args.iteration == 0 and args.pretrained_checkpoint is not None:
        original_load = args.load
        args.load = args.pretrained_checkpoint
        original_rng = args.no_load_rng
        args.no_load_rng = True
        _ = load_checkpoint(model, None, None)
        args.load = original_load
        args.no_load_rng = original_rng
        # This is critical when only model is loaded. We should make sure
        # main parameters are also updated.
        optimizer.reload_model_params()
    timers('pretrained checkpoint').stop()

    # Print setup timing.
    print_rank_last('done with setups ...')
    timers.log(['train/valid/test dataset/dataloder', 'callback function',
                'model and optimizer', 'pretrained checkpoint'], barrier=True)
    print_rank_last('training ...')

    # Finetune the model.
    if args.epochs > 0:
        _train(model, optimizer, opt_param_scheduler, forward_step,
               train_dataloader, valid_dataloader, end_of_epoch_callback, config)
    # Or just evaluate.
    else:
        if end_of_epoch_callback is not None:
            print_rank_last('evaluation only mode, setting epoch to -1')
            end_of_epoch_callback(model, epoch=-1, output_predictions=True)
    print_rank_last('done :-)')
