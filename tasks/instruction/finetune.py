"""Insturction Tuning."""
from functools import partial
from megatron import get_args
from megatron import print_rank_0, print_rank_last
from megatron import get_tokenizer
from megatron.model import GPTModel
from tasks.eval_utils import accuracy_func_provider
from tasks.finetune_utils import finetune
from tasks.instruction.data import InstructionDataset
from megatron import get_timers
from megatron.arguments import core_transformer_config_from_args
from megatron.utils import average_losses_across_data_parallel_group, get_ltor_masks_and_position_ids
from megatron.model.enums import AttnMaskType
import torch
from collections.abc import Iterable
import copy

def train_valid_datasets_provider():
    """Provide train and validation datasets."""
    args = get_args()
    tokenizer = get_tokenizer()
    train_data = []
    for path in args.train_data:
        try:
            weight = float(path)
            train_data.append([weight]) # weight
        except ValueError as e:
            train_data[-1].append(path) # name and path
    train_datasets = []
    for d in train_data:
        train_dataset = InstructionDataset(d[1], d[0], d[2:], tokenizer, args.seq_length,
                        args.use_mix_format, args.use_cache, args.rank)
        train_datasets.append(train_dataset)
    
    if args.valid_data is not None:
        valid_data = []
        for path in args.valid_data:
            try:
                weight = float(path)
                valid_data.append([weight]) # weight
            except ValueError as e:
                valid_data[-1].append(path) # name and path
        valid_datasets = []
        for d in valid_data:
            valid_dataset = InstructionDataset(d[1], d[0], d[2:], tokenizer, args.seq_length,
                            args.use_mix_format, args.use_cache, args.rank)
            valid_datasets.append(valid_dataset)
    else:
        valid_datasets = None
    
    return train_datasets, valid_datasets

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
        encoder_attn_mask_type=AttnMaskType.padding,
    )
    return model

def process_batch(batch):
    """Process batch and produce inputs for the model."""
    args = get_args()

    tokens = batch['input_ids'].long().cuda().contiguous()
    labels = batch['labels'].long().cuda().contiguous()
    attention_mask = batch['attention_mask'].unsqueeze(1).cuda().contiguous()
    position_ids = batch['position_ids'].cuda().contiguous()
    loss_mask = batch['loss_mask'].cuda().contiguous()
    assistant_loss_mask = batch['assistant_loss_mask'].cuda().contiguous()
    # tokenizer = get_tokenizer()
    # print_rank_0(tokens.shape)
    # position_id = position_ids[0].tolist()
    # print_rank_0(position_id[-13:])
    # print_rank_0(tokens[0, -13:])
    # print_rank_0(attention_mask[0, 0, -13:, -13:])
    # start_index = 0
    # cnt = 0
    # input_ids = tokens[0].tolist()
    # while start_index < tokens.shape[1]:
    #     try:
    #         end_index = position_id.index(0, start_index + 1)
    #     except ValueError:
    #         end_index = tokens.shape[1]
    #     if end_index - start_index == 1:
    #         break
    #     print_rank_0(f"sample {cnt + 1}:")
    #     print_rank_0(f"{tokenizer.detokenize(input_ids[start_index:end_index])}")
    #     start_index = end_index
    #     cnt += 1
    # print_rank_last(tokenizer.pad)
    # for i in range(tokens.shape[0]):
    #     print_rank_0(f"Input {i} from {dataset_name}:")
    #     print_rank_0(tokens.shape)
    #     print_rank_0(tokenizer.detokenize(tokens[i].tolist()))
    #     seq_len = torch.logical_not(attention_mask[i, 0, :, 0]).sum().item()
    #     print_rank_0(f"seq_len: {seq_len}")
    #     print_rank_0(f"theory attention sum: {seq_len * (seq_len + 1) / 2}")
    #     print_rank_0(f"attention_mask_sum: {torch.logical_not(attention_mask[i, 0, :, :]).sum()}")
    #     print_rank_0(f"loss_mask_sum: {loss_mask[i, :].sum()}")
    #     print_rank_0(f"loss_mask_sum(seq_len): {loss_mask[i, :seq_len].sum()}")
    #     print_rank_0(f"attention_mask: {attention_mask[i, 0, :seq_len + 1, :seq_len + 1]}")
    #     print_rank_0(f"loss_mask: {loss_mask[i, :seq_len + 1]}")
    # quit()

    return tokens, labels, position_ids, attention_mask, loss_mask, assistant_loss_mask


def cross_entropy_loss_func(loss_mask, assistant_loss_mask, dataset_name, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    assistant_loss_mask = assistant_loss_mask.view(-1).float()
    assistant_loss = torch.sum(losses.view(-1) * assistant_loss_mask) / assistant_loss_mask.sum()
    # loss = output_tensor.mean()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss, assistant_loss])
    # print(f"loss mask sum {loss_mask.sum()}, {loss}, {assistant_loss}")
    loss_dict = {}
    loss_dict['lm loss'] = averaged_loss[0]
    loss_dict['assistant loss'] = averaged_loss[1]
    if dataset_name is not None:
        loss_dict[f'lm loss for {dataset_name}'] = averaged_loss[0]
        loss_dict[f'assistant loss for {dataset_name}'] = averaged_loss[1]

    return loss, loss_dict

def _cross_entropy_forward_step(batch, model):
    """Simple forward step with cross-entropy loss."""
    timers = get_timers()
    # print_rank_0(type(batch))

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    if isinstance(batch, Iterable):
        iteraion, (dataset_name, batch_) = next(batch)
    else:
        batch_ = batch
        dataset_name = None
    tokens, labels, position_ids, attention_mask, loss_mask, assistant_loss_mask = process_batch(batch_)
    timers('batch-generator').stop()

    # Forward model.
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    
    return output_tensor, partial(cross_entropy_loss_func, loss_mask, assistant_loss_mask, dataset_name)

def main():
    print_rank_0("Instruction Tuning")
    args = get_args()
    finetune(train_valid_datasets_provider, model_provider,
             end_of_epoch_callback_provider=None,
             task_collate_fn=partial(InstructionDataset.collate_fn, args.seq_length, args.pad_to_max_length, args.user_loss_mask),
             forward_step=_cross_entropy_forward_step)