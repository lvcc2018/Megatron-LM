# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Pretrain utilities."""

import math
import sys
import time
import re
from datetime import datetime

import torch

from megatron import (get_args, get_current_global_batch_size,
                      get_num_microbatches, get_signal_handler,
                      get_tensorboard_writer, get_timers, is_last_rank,
                      print_rank_0, print_rank_last, update_num_microbatches, get_writer)
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.core import mpu, tensor_parallel
from megatron.core.utils import get_model_config
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.initialize import (initialize_megatron, set_jit_fusion_options,
                                 write_args_to_tensorboard)
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module, GPTModel
from megatron.model.utils import fix_model_params
from megatron import print_rank_0
from megatron import print_rank_last
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.model import Float16Module
from megatron.model import GPTModel
from megatron.core import DistributedDataParallel as DDP
from megatron.core.enums import ModelType
from megatron.optimizer import get_megatron_optimizer
from megatron.initialize import initialize_megatron
from megatron.initialize import write_args_to_tensorboard
from megatron.initialize import set_jit_fusion_options
from megatron.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import unwrap_model
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.utils import calc_params_l2_norm
from megatron.core.pipeline_parallel import finalize_model_grads, get_forward_backward_func
from megatron.utils import report_memory
from megatron.model.vision.knn_monitor import compute_feature_bank
from megatron.optimizer import get_megatron_optimizer
from megatron.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.utils import (calc_params_l2_norm,
                            check_adlr_autoresume_termination, report_memory,
                            unwrap_model)
from megatron.data.dataloader_manager import DataLoaderManager, ConstantRateScheduler
from megatron.data.dataset_manager import DatasetManager

# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             process_non_loss_data_func=None,
             extra_args_provider=None,
             args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    args = get_args()
    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type)
    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')
    config = get_model_config(model[0])

    # Data stuff.
    timers('train/valid/test-data-iterators-setup', log_level=0).start(
        barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        all_data_iterators = [
            build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
            for _ in range(len(model))
        ]
        train_data_iterator = [data_iterators[0]
                               for data_iterators in all_data_iterators]
        valid_data_iterator = [data_iterators[1]
                               for data_iterators in all_data_iterators]
        test_data_iterator = [data_iterators[2]
                              for data_iterators in all_data_iterators]
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup',
                'train/valid/test-data-iterators-setup'], barrier=True)

    if not args.skip_train:
        print_rank_0('training ...')

        if args.dataloader_type == 'cyclic' and args.retro_add_retriever:
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration = train(forward_step_func,
                              model, optimizer, opt_param_scheduler,
                              train_data_iterator, valid_data_iterator,
                              process_non_loss_data_func, config)

        print_datetime('after training is done')

        if args.save and iteration != 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration

    if args.do_valid:
        prefix = f'iteration {iteration} on validation set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)
    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
    writer = get_writer()
    if writer is not None:
        writer.finish()
    if args.do_test:
        prefix = f'iteration {iteration} on test set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)


def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // \
                      args.global_batch_size
        args.train_iters = iterations

    print_rank_0('setting training iterations to {}'.format(args.train_iters))


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       args.virtual_pipeline_model_parallel_size is not None:
        assert model_type != ModelType.encoder_and_decoder, \
            "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert args.pipeline_model_parallel_split_rank is not None, \
                    "Split rank needs to be specified for model with both encoder and decoder"
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (
                        rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder)
        else:
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Disallow training and inference with Transformer Engine
    # for non-GPT models
    args.allow_transformer_engine = all([type(m) == GPTModel for m in model])
    assert args.allow_transformer_engine or args.transformer_impl == 'local', \
        'Transformer Engine is only approved for GPT models'

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([sum([p.nelement() for p in model_module.parameters()])
                for model_module in model])), flush=True)

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if wrap_with_ddp:
        config = get_model_config(model[0])
        model = [DDP(config,
                     model_module,
                     data_parallel_group=mpu.get_data_parallel_group(),
                     accumulate_allreduce_grads_in_fp32=args.accumulate_allreduce_grads_in_fp32,
                     overlap_grad_reduce=args.overlap_grad_reduce,
                     use_distributed_optimizer=args.use_distributed_optimizer)
                 for model_module in model]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model


def get_optimizer_param_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        lr_decay_steps = args.lr_decay_iters * args.global_batch_size
        wd_incr_steps = args.train_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        lr_decay_steps = args.lr_decay_samples
        wd_incr_steps = args.train_samples
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler)

    return opt_param_scheduler


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func, model_type)
    if args.fixed_params or args.trainable_params:
        fix_model_params(model,
                         fixed_params=args.fixed_params,
                         trainable_params=args.trainable_params,
                         keep_res_trainable=args.keep_res_trainable)
    unwrapped_model = unwrap_model(model)

    # Not load optim or rng, set up optimizer and rng before loading
    # Deal with the case of continue training from a checkpoint
    optimizer = get_megatron_optimizer(
        model, no_wd_decay_cond, scale_lr_cond, lr_mult
    ) if not args.no_load_optim else None
    opt_param_scheduler = get_optimizer_param_scheduler(
        optimizer) if not args.no_load_optim else None

    if args.load is not None:
        timers = get_timers()
        timers('load-checkpoint', log_level=0).start(barrier=True)
        args.iteration = load_checkpoint(model, optimizer, opt_param_scheduler)
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
    else:
        args.iteration = 0

    optimizer = get_megatron_optimizer(
        model, no_wd_decay_cond, scale_lr_cond, lr_mult
    ) if args.no_load_optim else optimizer
    opt_param_scheduler = get_optimizer_param_scheduler(
        optimizer) if args.no_load_optim else opt_param_scheduler

    # We only support local DDP with multiple micro-batches.
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == 'local'

    # get model without FP16 and/or DDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, opt_param_scheduler



def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Set grad to zero.
    for partition in model:
        partition.zero_grad_buffer()
    optimizer.zero_grad()

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False)

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
    timers('optimizer').stop()

    # Gather params.
    if update_successful:
        optimizer.gather_model_params(args, timers)

    # Vision momentum.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0]:
            if "consumed samples" in key and args.use_dataset_manager:
                loss_reduced[key] = sum([x[key] for x in losses_reduced])
            else:
                losses_reduced_for_key = [x[key] for x in losses_reduced if x[key] > 0]
                if len(losses_reduced_for_key) > 0:
                    loss_reduced[key] = sum(
                        losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad,
                 epoch_states=None, dataset_iters=None):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    # writer = get_tensorboard_writer()
    writer = get_writer()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
            # record dataset name for log
            if dataset_iters is not None:
                dataset_iters[key] = dataset_iters.get(key, 0) + 1
        else:
            # value = loss_dict[key].float().sum().item()
            value = loss_dict[key]
            is_nan = value == float('inf') or \
                value == -float('inf') or \
                value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'all-grads-sync',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar('learning-rate', learning_rate, iteration)
            writer.add_scalar('learning-rate vs samples', learning_rate,
                              args.consumed_train_samples, only_tensorboard=True)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar('batch-size', batch_size, iteration)
            writer.add_scalar('batch-size vs samples', batch_size,
                              args.consumed_train_samples, only_tensorboard=True)
        for key in loss_dict:
            writer.add_scalar(f"loss/{key}", loss_dict[key], iteration)
            writer.add_scalar(f"loss/{key} vs samples", loss_dict[key],
                              args.consumed_train_samples, only_tensorboard=True)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale vs samples', loss_scale,
                              args.consumed_train_samples, only_tensorboard=True)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar('world-size', args.world_size, iteration)
            writer.add_scalar('world-size vs samples', args.world_size,
                              args.consumed_train_samples, only_tensorboard=True)
        if grad_norm is not None:
            writer.add_scalar('grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm vs samples', grad_norm,
                              args.consumed_train_samples, only_tensorboard=True)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples, only_tensorboard=True)
        if params_norm is not None:
            writer.add_scalar('params-norm', params_norm, iteration)
            writer.add_scalar('params-norm vs samples', params_norm,
                              args.consumed_train_samples, only_tensorboard=True)
        if epoch_states is not None:
            for key in epoch_states:
                writer.add_scalar(f'epoch/{key}', epoch_states[key], iteration)
        if args.use_dataloader_manager or args.use_dataset_manager:
            for key in args.consumed_train_samples_per_dataset:
                writer.add_scalar(f'consumed_samples/{key}',
                                  args.consumed_train_samples_per_dataset[key],
                                  iteration)
                writer.add_scalar(f'consumed_samples/{key} vs samples',
                                  args.consumed_train_samples_per_dataset[key],
                                  args.consumed_train_samples, only_tensorboard=True)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations
        if writer:
            if args.log_timers_to_tensorboard:
                writer.add_scalar('iteration-time',
                                  elapsed_time_per_iteration, iteration)
        log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        if args.use_dataloader_manager or args.use_dataset_manager:
            for key in args.consumed_train_samples_per_dataset:
                log_string += ' consumed samples of {:s}: {:12d} |'.format(
                    key, args.consumed_train_samples_per_dataset[key])
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        checkpoint_activations_factor = 4 if args.recompute_granularity is not None else 3
        ffn_factor = 2 if args.swiglu else 1
        seq_len = args.seq_length
        hidden_size = args.hidden_size
        ffn_hidden_size = args.ffn_hidden_size
        num_layers = args.num_layers
        vocab_size = args.padded_vocab_size
        flops_per_iteration = \
            checkpoint_activations_factor * \
                (   \
                    num_layers * \
                    (   \
                        8 * batch_size * seq_len * hidden_size ** 2 + \
                        4 * batch_size * seq_len ** 2 * hidden_size + \
                        2 * batch_size * seq_len * hidden_size * ffn_hidden_size * (ffn_factor + 1)
                    ) + \
                    (   \
                    2 * batch_size * seq_len * hidden_size * vocab_size \
                    )   \
                )
        tflops = flops_per_iteration / (elapsed_time_per_iteration * args.world_size * (10**12))
        log_string += ' TFLOPS per GPU: {:3f} |'.format(
            tflops)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                if dataset_iters is None:
                    avg = total_loss_dict[key].item() / \
                        float(max(1, total_loss_dict[advanced_iters_key]))
                else:
                    # log for each dataset source
                    iters = dataset_iters.get(key, 0)
                    if iters == 0:
                        avg = 0
                    else:
                        avg = total_loss_dict[key].item(
                        ) / float(max(1, iters))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        if epoch_states is not None:
            for key in epoch_states:
                log_string += ' epoch of {:s}: {:.3f} |'.format(
                    key, epoch_states[key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        if dataset_iters is not None:
            # reset iters for each dataset
            for key in dataset_iters:
                dataset_iters[key] = 0
        print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler):
    timers = get_timers()
    # Extra barrier is added to make sure
    # all ranks report the max time.
    timers('save-checkpoint', log_level=0).start(barrier=True)
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
    timers('save-checkpoint').stop(barrier=True)
    timers.log(['save-checkpoint'])


def train(forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func, config):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}
    if args.use_dataloader_manager or args.use_dataset_manager:
        total_dataset_iters = {}
    else:
        total_dataset_iters = None

    # Iterations.
    iteration = args.iteration

    # Setup some training config params
    config.grad_scale_func = optimizer.scale_loss
    config.timers = timers
    # TODO: Remove this once we move DDP to Core.
    if len(model) == 1 and isinstance(model[0], DDP) and \
        args.overlap_grad_reduce:
        assert config.no_sync_func is None, \
            ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
             'a custom no_sync_func is not supported when overlapping grad-reduce')
        if args.delay_grad_reduce:
            config.grad_sync_func = model[0].grad_sync
        config.no_sync_func = model[0].no_sync
    config.finalize_model_grads_func = finalize_model_grads

    timers('interval-time', log_level=0).start(barrier=True)
    print_datetime('before the start of training step')
    report_memory_flag = True
    while iteration < args.train_iters:
        if args.profile and \
           iteration == args.profile_step_start and \
           torch.distributed.get_rank() in args.profile_ranks:
            torch.cuda.cudart().cudaProfilerStart()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        update_num_microbatches(args.consumed_train_samples)
        args.curr_iteration = iteration
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                       train_data_iterator,
                       model,
                       optimizer,
                       opt_param_scheduler,
                       config)
        iteration += 1
        args.consumed_train_samples += mpu.get_data_parallel_world_size() * \
            args.micro_batch_size * \
            get_num_microbatches()
        # record consumed samples in each dataset
        if args.use_dataloader_manager:
            name = args.current_dataset_name
            if name not in args.consumed_train_samples_per_dataset:
                args.consumed_train_samples_per_dataset[name] = 0
            args.consumed_train_samples_per_dataset[name] += mpu.get_data_parallel_world_size() * \
                                                                args.micro_batch_size * \
                                                                get_num_microbatches()
        if args.use_dataset_manager:
            for key in args.dataset_names:
                consumed_samples = loss_dict.pop(f"consumed samples for {key}", 0)
                if key not in args.consumed_train_samples_per_dataset:
                    args.consumed_train_samples_per_dataset[key] = 0
                args.consumed_train_samples_per_dataset[key] += consumed_samples

        # Logging.
        loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, num_zeros_in_grad,
                                          dataset_iters=total_dataset_iters)

        # Autoresume
        if args.adlr_autoresume and \
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              opt_param_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
           args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(prefix, forward_step_func,
                                       valid_data_iterator, model,
                                       iteration, process_non_loss_data_func,
                                       config, False)

        # Checkpointing
        saved_checkpoint = False
        if args.exit_signal_handler:
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler)
                print_datetime('exiting program after receiving SIGTERM.')
                sys.exit()

        if args.save and args.save_interval and \
           iteration % args.save_interval == 0:
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler)
            saved_checkpoint = True

        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.cuda.IntTensor(
                [train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             opt_param_scheduler)
                print_datetime('exiting program after {} minutes'.format(train_time))
                sys.exit()

        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            sys.exit()

        if args.profile and \
           iteration == args.profile_step_end and \
           torch.distributed.get_rank() in args.profile_ranks:
            torch.cuda.cudart().cudaProfilerStop()

    return iteration


def evaluate(forward_step_func,
             data_iterator,
             model,
             process_non_loss_data_func,
             config,
             verbose=False):
    """Evaluation."""
    args = get_args()

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    total_loss_dict = {}
    total_eval_iters = {}

    # make validation batch size independent from training batch size
    eval_batch_size = args.global_batch_size
    eval_num_microbatches = eval_batch_size // \
        (args.micro_batch_size * args.data_parallel_size)

    with torch.no_grad():
        iteration = 0
        if verbose:
            print_rank_0(f'Evaluating on {args.eval_iters * eval_batch_size} samples')
        while iteration < args.eval_iters:
            iteration += 1
            if verbose:
                print_rank_0(f'Evaluating iter {iteration}/{args.eval_iters}')

            forward_backward_func = get_forward_backward_func()
            # Don't care about timing during evaluation
            config.timers = None
            loss_dicts = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=eval_num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True)
            config.timers = get_timers()
            
            if args.use_dataset_manager:
                for key in args.dataset_names:
                    for loss_dict in loss_dicts:
                        consumed_samples = loss_dict.pop(f"consumed samples for {key}", 0)
                        if key not in args.consumed_valid_samples_per_dataset:
                            args.consumed_valid_samples_per_dataset[key] = 0
                        args.consumed_valid_samples_per_dataset[key] += consumed_samples
                

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        if loss_dict[key] > 0:
                            total_loss_dict[key] = total_loss_dict.get(
                                key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
                            total_eval_iters[key] = total_eval_iters.get(
                                key, 0) + 1

            args.consumed_valid_samples += eval_batch_size
            # record consumed samples in each dataset
            if args.use_dataloader_manager:
                name = args.current_dataset_name
                if name not in args.consumed_valid_samples_per_dataset:
                    args.consumed_valid_samples_per_dataset[name] = 0
                args.consumed_valid_samples_per_dataset[name] += mpu.get_data_parallel_world_size() * \
                                                                    args.micro_batch_size * \
                                                                    get_num_microbatches()
            
        
        collected_non_loss_data = None
        if process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
                collect_non_loss_data=True)

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        # total_loss_dict[key] /= args.eval_iters * get_num_microbatches()
        total_loss_dict[key] /= total_eval_iters[key]

    return total_loss_dict, collected_non_loss_data

def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, process_non_loss_data_func, config,
                               verbose=False, write_to_tensorboard=True):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    # writer = get_tensorboard_writer()
    writer = get_writer()

    total_loss_dict, collected_non_loss_data = evaluate(
        forward_step_func, data_iterator, model,
        process_non_loss_data_func, config, verbose)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer:
            # writer.add_scalar('{} validation'.format(key),
            #                   total_loss_dict[key].item(),
            #                   iteration)
            # writer.add_scalar('{} validation vs samples'.format(key),
            #                   total_loss_dict[key].item(),
            #                   args.consumed_train_samples)
            # if args.log_validation_ppl_to_tensorboard:
            #     writer.add_scalar('{} validation ppl'.format(key), ppl,
            #                       iteration)
            #     writer.add_scalar('{} validation ppl vs samples'.format(key),
            #                       ppl, args.consumed_train_samples)
            writer.add_scalar('validation/{}'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('validation/{} vs samples'.format(key),
                              total_loss_dict[key].item(),
                              args.consumed_train_samples, only_tensorboard=True)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('validation ppl/{}'.format(key), ppl,
                                  iteration)
                writer.add_scalar('validation ppl/{} vs samples'.format(key),
                                  ppl, args.consumed_train_samples, only_tensorboard=True)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
    """Build pretraining datasets."""

    args = get_args()

    # Number of train/valid/test samples.
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * \
                 args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [train_samples,
                                  eval_iters * args.global_batch_size,
                                  test_iters * args.global_batch_size]
    print_rank_0(' > datasets target sizes (minimum size):')
    print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
    print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
    print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

    # Build the datasets.
    return build_train_valid_test_datasets_provider(train_val_test_num_samples)


def build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider):
    """Build pretraining data loaders."""

    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_tensor_model_parallel_rank() == 0:

        # Build datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            build_train_valid_test_datasets_provider)
        if args.use_dataloader_manager:
            # get weights
            ds_weights = train_ds[1]
            ds_names = train_ds[2]

            # Build dataloders.
            train_dataloaders = []
            valid_dataloaders = []
            test_dataloaders = []
            for train_d_index, train_d in enumerate(train_ds[0]):
                train_d_name = ds_names[train_d_index]
                consumed_train_samples = args.consumed_train_samples_per_dataset.get(train_d_name, 0)
                train_dataloader = build_pretraining_data_loader(
                    train_d, consumed_train_samples) # Warning: arg consumed_train_samples may not be proper used here.
                train_dataloaders.append(train_dataloader)
            for valid_d_index, valid_d in enumerate(valid_ds[0]):
                valid_d_name = ds_names[valid_d_index]
                consumed_valid_samples = args.consumed_valid_samples_per_dataset.get(valid_d_name, 0)
                valid_dataloader = build_pretraining_data_loader(
                    valid_d, consumed_valid_samples) # Warning: arg consumed_train_samples may not be proper used here.
                valid_dataloaders.append(valid_dataloader)
            for test_d in test_ds[0]:
                test_dataloader = build_pretraining_data_loader(test_d, 0)
                test_dataloaders.append(test_dataloader)
            if len(train_dataloaders) == 0:
                train_dataloaders = None
            if len(valid_dataloaders) == 0:
                valid_dataloaders = None
            if len(test_dataloaders) == 0:
                test_dataloaders = None
            # print("rank {}: {}, {}, {}".format(args.rank, len(train_dataloaders), len(ds_weights), ds_weights))
            # Wrap dataloaders with DataLoaderManager
            if train_dataloaders is not None:
                consumed_train_iterations = args.consumed_train_samples // args.global_batch_size
                data_loaders = [(train_dataloaders[i], args.seed, ds_names[i], ConstantRateScheduler(ds_weights[i])) for i in range(len(train_dataloaders))]
                train_dataloader = DataLoaderManager(data_loaders,
                                                     rank=args.rank,
                                                     world_size=args.world_size, gradient_accumulation_steps=get_num_microbatches(),
                                                     seed=args.seed,
                                                     sample=True,
                                                     consumed_iterations=consumed_train_iterations)
            else:
                train_dataloader = None
            if valid_dataloaders is not None:
                consumed_valid_iterations = args.consumed_valid_samples // args.global_batch_size
                data_loaders = [(valid_dataloaders[i], args.seed, ds_names[i], ConstantRateScheduler(ds_weights[i])) for i in range(len(valid_dataloaders))]
                valid_dataloader = DataLoaderManager(data_loaders,
                                                     rank=args.rank,
                                                     world_size=args.world_size,
                                                     gradient_accumulation_steps=get_num_microbatches(),
                                                     seed=args.seed,
                                                     sample=True,
                                                     consumed_iterations=consumed_valid_iterations)
            else:
                valid_dataloader = None
            if test_dataloaders is not None:
                data_loaders = [(test_dataloaders[i], args.seed, ds_names[i], ConstantRateScheduler(ds_weights[i])) for i in range(len(test_dataloaders))]
                test_dataloader = DataLoaderManager(data_loaders,
                                                    rank=args.rank,
                                                    world_size=args.world_size,
                                                    gradient_accumulation_steps=get_num_microbatches(),
                                                    seed=args.seed, 
                                                    sample=True,
                                                    consumed_iterations=0)
            else:
                test_dataloader = None
        else:

            # Build dataloders.
            train_dataloader = build_pretraining_data_loader(
                train_ds, args.consumed_train_samples)
            if args.skip_train:
                valid_dataloader = build_pretraining_data_loader(valid_ds, 0)
            else:
                valid_dataloader = build_pretraining_data_loader(
                    valid_ds, args.consumed_valid_samples)
            test_dataloader = build_pretraining_data_loader(test_ds, 0)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()
    if args.use_dataset_manager:
        # broadcast dataset names
        if mpu.get_tensor_model_parallel_rank() == 0:
            dataset_names = [train_ds.names]
        else:
            dataset_names = [None]
        torch.distributed.broadcast_object_list(dataset_names, mpu.get_tensor_model_parallel_src_rank(), mpu.get_tensor_model_parallel_group())
        args.dataset_names = dataset_names[0]
    if args.use_dataloader_manager:
        # broadcast dataset names
        if mpu.get_tensor_model_parallel_rank() == 0:
            dataset_names = [train_ds[2]]
        else:
            dataset_names = [None]
        torch.distributed.broadcast_object_list(dataset_names, mpu.get_tensor_model_parallel_src_rank(), mpu.get_tensor_model_parallel_group())
        args.dataset_names = dataset_names[0]

    return train_dataloader, valid_dataloader, test_dataloader


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """Build pretraining data iterators."""

    args = get_args()

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = \
        build_train_valid_test_data_loaders(
            build_train_valid_test_datasets_provider)

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic']

    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(train_dataloader))
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(valid_dataloader))
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader) if dl_type == 'single' \
                             else iter(cyclic_iter(test_dataloader))
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator
