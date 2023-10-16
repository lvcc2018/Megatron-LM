# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

import torch
from collections import Counter
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel, mpu
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text', 'loss_mask']
    datatype = torch.int64

    # Broadcast data.
    dataset_name = ""
    source = [None]
    # print(f"rank {args.rank}: data_iterator is {data_iterator}")
    if data_iterator is not None:
        data = next(data_iterator)
        if args.use_dataset_manager:
            source = [data['source']]
        if args.use_dataloader_manager:
            dataset_name, data = data
    else:
        data = None
    # print(f"rank {args.rank}: data is {data}")
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    dataset_name = [dataset_name]
    torch.distributed.broadcast_object_list(dataset_name, src=mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
    dataset_name = dataset_name[0]
    torch.distributed.broadcast_object_list(source, src=mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
    source = source[0]
    if dataset_name != '':
        args.current_dataset_name = dataset_name

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    loss_mask = data_b['loss_mask'].contiguous()
        

    # Get the masks and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids, dataset_name, source

def loss_func(loss_mask, dataloader_manager_args, dataset_manager_args, output_tensor):
    use_dataloader_manager, dataset_name = dataloader_manager_args
    use_dataset_manager, source, all_names = dataset_manager_args
    losses = output_tensor.float()
    if use_dataset_manager:
        from megatron import print_rank_last
        loss_mask = loss_mask.float()
        each_loss = torch.sum(losses * loss_mask, dim=1) / loss_mask.sum(dim=1)
        tensor_list = [torch.zeros_like(each_loss) for _ in range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather(tensor_list, each_loss, group=mpu.get_data_parallel_group())
        source_list = [None] * mpu.get_data_parallel_world_size()
        torch.distributed.all_gather_object(source_list, source, group=mpu.get_data_parallel_group())
        source_loss = {}
        flatten_source_list = []
        for sources in source_list:
            flatten_source_list.extend(sources)
        tensor_list = torch.cat(tensor_list, dim=0)
        for each_loss, each_source in zip(tensor_list, flatten_source_list):
            if each_source not in source_loss:
                source_loss[each_source] = []
            source_loss[each_source].append(each_loss)
        for key, value in source_loss.items():
            source_loss[key] = torch.stack(value).mean().item()
        for key in all_names:
            if key not in source_loss:
                source_loss[key] = -1
        consumed_source_number = Counter(flatten_source_list)
    
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss_dict = {
        'lm loss': averaged_loss[0].item()
    }
    if use_dataloader_manager and dataset_name != "":
        loss_dict[f'lm loss for {dataset_name}'] = averaged_loss[0]
    if use_dataset_manager:
        for key, value in source_loss.items():
            loss_dict[f'lm loss for {key}'] = value
        for key, value in consumed_source_number.items():
            loss_dict[f"consumed samples for {key}"] = value
        for key in all_names:
            if f"consumed samples for {key}" not in loss_dict:
                loss_dict[f"consumed samples for {key}"] = 0
    return loss, loss_dict


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, dataset_name, source = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)
    dataloader_manager_args = (args.use_dataloader_manager, dataset_name)
    if args.use_dataset_manager:
        dataset_manager_args = (args.use_dataset_manager, source, args.dataset_names)
    else:
        dataset_manager_args = (args.use_dataset_manager, None, None)

    return output_tensor, partial(loss_func, loss_mask, dataloader_manager_args, dataset_manager_args)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        use_dataloader_manager=args.use_dataloader_manager,
        use_dataset_manager=args.use_dataset_manager,
        global_batch_size=args.global_batch_size,
        data_cache_path=args.data_cache_path)
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step)
