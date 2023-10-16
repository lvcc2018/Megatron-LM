# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from collections import Counter
from torch import Tensor
from functools import partial
from typing import Union
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel, mpu
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import GPTDataset, build_train_valid_test_datasets
import megatron.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.transformer.spec_utils import import_module
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import (
    gpt_layer_with_transformer_engine_spec,
    gpt_layer_with_transformer_engine_spec_moe
)

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.model.GPTModel]: The returned model
    """
    args = get_args()

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())

    if args.use_mcore_models:
        if args.model_spec is not None:
            transformer_layer_spec = import_module(args.model_spec)
        else:
            if args.num_experts is None:
                transformer_layer_spec = gpt_layer_with_transformer_engine_spec
            else:
                transformer_layer_spec = gpt_layer_with_transformer_engine_spec_moe

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent
        )
    else:
        model = megatron.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )

    return model


def get_batch(data_iterator):
    """Generate a batch."""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text', 'loss_mask']
    datatype = torch.int64

    # Broadcast data.
    dataset_name = ""
    source = None
    # print(f"rank {args.rank}: data_iterator is {data_iterator}")
    if data_iterator is not None:
        data = next(data_iterator)
        if args.use_dataloader_manager:
            dataset_name, data = data
    else:
        data = None
    # print(f"rank {args.rank}: data is {data}")
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    dataset_name = [dataset_name]
    torch.distributed.broadcast_object_list(dataset_name, src=mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
    dataset_name = dataset_name[0]
    if dataset_name != '':
        args.current_dataset_name = dataset_name
    if args.use_dataset_manager:
        source = tensor_parallel.broadcast_data(['source'], data, torch.int64)['source']

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
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
    """    
    use_dataloader_manager, dataset_name = dataloader_manager_args
    use_dataset_manager, source, all_names = dataset_manager_args
    losses = output_tensor.float()
    if use_dataset_manager:
        with torch.no_grad():
            from megatron import print_rank_last
            loss_mask = loss_mask.float()
            each_loss = torch.sum(losses * loss_mask, dim=1) / loss_mask.sum(dim=1)
            tensor_list = [torch.zeros_like(each_loss) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(tensor_list, each_loss, group=mpu.get_data_parallel_group())
            source_list = [torch.zeros_like(source) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(source_list, source, group=mpu.get_data_parallel_group())
            source_loss = {}
            flatten_source_list = torch.cat(source_list, dim=0).view(-1).tolist()
            flatten_source_list = [all_names[s] for s in flatten_source_list]
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

    # Check individual rank losses are not NaN prior to DP all-reduce.
    args = get_args()
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

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


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
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
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
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
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
