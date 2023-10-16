import argparse
import os
import time
import sys
# append megatron
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from megatron.tokenizer.tokenizer import _GPTSentencePieceTokenizer, _LLaMaSentencePieceTokenizer
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.core import mpu
import torch
import numpy as np
from tqdm import tqdm
import json
from datetime import timedelta

def init_parser():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("data")
    group.add_argument('--data-path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ... It is used with --split when a '
                       'single dataset used for all three: train, valid '
                       'and test. It is exclusive to the other '
                       '--*-data-path args')
    group.add_argument('--data-impl', type=str, default='infer',
                       choices=['mmap', 'infer'],
                       help='Implementation of indexed datasets.')
    group.add_argument('--split', type=str, default='969, 30, 1',
                       help='Comma-separated list of proportions for training,'
                       ' validation, and test split. For example the split '
                       '`90,5,5` will use 90%% of data for training, 5%% for '
                       'validation and 5%% for test.')
    group.add_argument('--seq-length', type=int, default=None,
                       help='Maximum sequence length to process.')
    group.add_argument('--seed', type=int, default=1234,
                       help='Random seed used for python, numpy, '
                       'pytorch, and cuda.')
    group.add_argument('--mmap-warmup', action='store_true',
                       help='Warm up mmap files.')
    group.add_argument('--train-data-path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--valid-data-path', nargs='*', default=None,
                       help='Path to the validation dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--test-data-path', nargs='*', default=None,
                       help='Path to the test dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--data-cache-path', default=None,
                       help='Path to a directory to hold cached index files.')
    group.add_argument('--train-samples', type=int, default=None,
                       help='Total number of samples to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--consumed-samples', type=int, default=0)
    group.add_argument('--output-file', type=str, default=None,
                       help="directory for output.")
    group.add_argument('--tokenizer-type', type=str,
                       default=None,
                       choices=['GPTSentencePieceTokenizer',
                                'LLaMaSentencePieceTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='Sentencepiece tokenizer model.')
    return parser

def fake_distributed_initialize(args):
    device_count = 1
    # Call the init process
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12000"
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=1,
        rank=0,
        timeout=timedelta(minutes=10),
    )

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(
                1,
                1,
            )
            print(
                f"> initialized tensor model parallel with size "
                f"{mpu.get_tensor_model_parallel_world_size()}"
            )
            print(
                f"> initialized pipeline model parallel with size "
                f"{mpu.get_pipeline_model_parallel_world_size()}"
            )

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    fake_distributed_initialize(args)
    # Number of train/valid/test samples.
    train_samples = args.train_samples
    train_val_test_num_samples = [train_samples, 0, 0]
    print(' > datasets target sizes (minimum size):')
    print('    train:      {}'.format(train_val_test_num_samples[0]))
    print('    validation: {}'.format(train_val_test_num_samples[1]))
    print('    test:       {}'.format(train_val_test_num_samples[2]))
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
        use_dataloader_manager=False,
        use_dataset_manager=False,
        global_batch_size=None,
        data_cache_path=args.data_cache_path,
        return_doc_ids=True)
    if args.tokenizer_type == "GPTSentencePieceTokenizer":
        tokenizer = _GPTSentencePieceTokenizer(args.tokenizer_model)
    elif args.tokenizer_type == "LLaMaSentencePieceTokenizer":
        tokenizer = _LLaMaSentencePieceTokenizer(args.tokenizer_model)
    else:
        raise NotImplementedError(f"Tokenizer type should be one of [GPTSentencePieceTokenizer, LLaMaSentencePieceTokenizer]")
    consumed_docs = {}
    eos_id = tokenizer._eos_id
    eos_token = tokenizer.detokenize([eos_id])
    print(f"eos id is {eos_id} and eos token is {eos_token}")
    sizes = train_ds.indexed_dataset.sizes
    
    for i in tqdm(range(args.consumed_samples), total=args.consumed_samples, desc="Counting"):
        d = train_ds[i]
        doc_ids = d['doc_ids'].tolist()
        loss_mask = d['loss_mask']
        text = d['text'][:-1]
        sample_tokens = np.sum(loss_mask).item()
        if len(doc_ids) > 2:
            # full document is in current sample
            for doc_id in doc_ids[1:-1]:
                if doc_id not in consumed_docs:
                    consumed_docs[doc_id] = 0
                consumed_docs[doc_id] += 1
        # cal first
        first_doc_id = doc_ids[0]
        doc_ends_index = np.where(text == eos_id)[0].tolist()
        if len(doc_ends_index) > 0:
            first_doc_end = doc_ends_index[0] + 1
        else:
            assert len(doc_ids) == 1, f"sample {i} have multiple docs while have no eos id"
            first_doc_end = sample_tokens
        size = sizes[first_doc_id]
        if first_doc_id not in consumed_docs:
            consumed_docs[first_doc_id] = 0
        consumed_docs[first_doc_id] += first_doc_end / size
        if len(doc_ids) == 1:
            continue
        # cal last
        last_doc_id = doc_ids[-1]
        last_doc_begin = doc_ends_index[-1] + 1
        size = sizes[last_doc_id]
        if last_doc_id not in consumed_docs:
            consumed_docs[last_doc_id] = 0
        consumed_docs[last_doc_id] += ((sample_tokens - last_doc_begin) / size)
    more_than_one = 0
    for doc_id in consumed_docs:
        train_num = consumed_docs[doc_id]
        if train_num > 1:
            more_than_one += 1
    print(f"docs number is {len(sizes)}, consume {len(consumed_docs)} docs ({len(consumed_docs) / len(sizes):.4f}), consume {more_than_one} ({more_than_one / len(sizes)}) docs more than once")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(consumed_docs, f, indent=4, ensure_ascii=False)
    print(f"write consumed information into {args.output_file}")
        

        
        
    