import argparse
import os
import time
import sys
import numpy as np
# append megatron
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from megatron.tokenizer import build_tokenizer
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.core import mpu
import torch
from datetime import timedelta




def init_args():
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
    group.add_argument('--use-dataloader-manager', action='store_true',
                       help='Whether use dataloader manager to sample data.')
    group.add_argument('--use-dataset-manager', action='store_true',
                       help='Whether use dataloader manager to sample data.')
    group.add_argument('--global-batch-size', type=int, default=None,
                       help='Training batch size. If set, it should be a '
                       'multiple of micro-batch-size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro-batch-size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')
    group.add_argument('--train-samples', type=int, default=None,
                       help='Total number of samples to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--train-iters', type=int, default=None,
                       help='Total number of iterations to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--eval-iters', type=int, default=100,
                       help='Number of iterations to run for evaluation'
                       'validation/test for.')
    group.add_argument('--eval-interval', type=int, default=1000,
                       help='Interval between running evaluation on '
                       'validation set.')
    group.add_argument('--iteration-start', type=int, default=None,
                       help="begining iteration number")
    group.add_argument('--iteration-end', type=int, default=None,
                       help="end iteration for print")
    group.add_argument('--index-start', type=int, default=None,
                       help="start index to print")
    group.add_argument('--index-end', type=int, default=None,
                       help="end index to print")
    group.add_argument('--output-dir', type=str, default=None,
                       help="directory for output.")
    group.add_argument('--tokenizer-type', type=str,
                       default=None,
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer',
                                'SentencePieceTokenizer',
                                'MixedTokenizer',
                                'GPTSentencePieceTokenizer',
                                'UL2SentencePieceTokenizer',
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--tokenizer-file', type=str, default=None,
                       help='BPE tokenizer file.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--vocab-size', default=786,
                       help='size of vocab for use with NullTokenizer')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--doc-ids', type=str, default=None,
                       help='The ids of the documents to view. Comma-separated list of ids, or ranges (e.g. 1-5). For example, `2,4,6,8,10` or `1-5,10-15,20,21`.')     
    group.add_argument('--show-iters', action='store_true',
                       help='Also show the corresponding iterations of each doc.')              
    args = parser.parse_args()
    args.vocab_extra_ids = 0
    # default settings
    args.make_vocab_size_divisible_by = 1 
    args.rank = 1
    return args

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
        args.tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
        args.pipeline_model_parallels_size = mpu.get_pipeline_model_parallel_world_size()


if __name__ == "__main__":
    args = init_args()
    fake_distributed_initialize(args)
    # Number of train/valid/test samples.
    if args.train_samples:
        train_samples = args.train_samples
        assert args.train_iters is None
    else:
        train_samples = args.train_iters * args.global_batch_size
        assert args.train_samples is None
    eval_iters = (args.train_iters // args.eval_interval + 1) * \
                 args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [train_samples,
                                  eval_iters * args.global_batch_size,
                                  test_iters * args.global_batch_size]
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
        use_dataloader_manager=args.use_dataloader_manager,
        use_dataset_manager=args.use_dataset_manager,
        global_batch_size=args.global_batch_size,
        data_cache_path=args.data_cache_path,
        return_doc_ids=True)
    iteration_set_flag = True
    doc_ids_set_flag = True
    if args.iteration_start is not None:
        assert args.index_start is None, f"iteration-start and index-start should only set one."
        if args.iteration_end is None:
            args.iteration_end = args.iteration_start
        args.index_start = (args.iteration_start - 1) * args.global_batch_size
        args.index_end = args.iteration_end * args.global_batch_size - 1
    elif args.index_start is not None:
        if args.index_end is None:
            args.index_end = args.index_start
    else:
        iteration_set_flag = False
    if args.doc_ids is None:
        doc_ids_set_flag = False
    else:
        def parse_doc_ids(doc_ids_str):
            doc_ids = []
            for part in doc_ids_str.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    doc_ids.extend(range(start, end + 1))
                else:
                    doc_ids.append(int(part))
            return doc_ids
        doc_ids = parse_doc_ids(args.doc_ids)
    if not iteration_set_flag and not doc_ids_set_flag:
        raise ValueError("You should set index-start or iteration-start or doc-ids")
    output_dir = args.output_dir
    assert output_dir is not None, "You should set --output-dir"
    tokenizer = build_tokenizer(args)
    os.makedirs(output_dir, exist_ok=True)
    if iteration_set_flag:  
        filename_format = "iteration-{:06d}.txt"
        prev_iteration = None
        fout = None
        start_time = time.time()
        for i in range(args.index_start, args.index_end + 1):
            iteration = i // args.global_batch_size + 1
            if iteration != prev_iteration:
                filename = filename_format.format(iteration)
                filepath = os.path.join(output_dir, filename)
                if fout is not None:
                    fout.close()
                fout = open(filepath, 'w', encoding='utf-8')
                prev_iteration = iteration
                print(f"To show data in iteration {iteration} to {filepath}, elapsed time: {time.time() - start_time}")
            print("Index:", i, "Iteration:", iteration, file=fout)
            data = train_ds[i]
            text = data['text']
            loss_mask = data['loss_mask']
            source = data.get('source', None)
            loss_sum = loss_mask.sum().item()
            text_length = (text != 0).sum().item()
            assert loss_sum <= args.seq_length, f"Index {i} loss mask {loss_sum} is longer than ${args.seq_length}"
            assert loss_sum == (text_length - 1), f"Index {i} loss mask {loss_sum} is not equal to text length {text_length - 1}"
            print("Doc ids:", data["doc_ids"], file=fout)
            if source is not None:
                print("Source:", source, file=fout)
            print("loss mask:", loss_mask.sum().item(), file=fout)
            print("tokens:", (text != 0).sum().item(), file=fout)
            print("=" * 80, file=fout)
            print(tokenizer.detokenize(text.tolist()), file=fout)
            print("=" * 80, file=fout)
        fout.close()
    if doc_ids_set_flag:
        filename_format = "doc-{:09d}.txt"
        prev_doc_id = None
        fout = None
        start_time = time.time()
        for doc_id in doc_ids:
            data = train_ds.indexed_dataset.get(doc_id)
            text = np.array(data, dtype=np.int64)
            filename = filename_format.format(doc_id)
            filepath = os.path.join(output_dir, filename)
            if fout is not None:
                fout.close()
            fout = open(filepath, 'w', encoding='utf-8')
            print(f"To show data in doc id {doc_id} to {filepath}, elapsed time: {time.time() - start_time}")
            print("Doc id:", doc_id, file=fout)
            print("tokens:", (text != 0).sum().item(), file=fout)
            if args.show_iters:
                indexs = train_ds.doc_id_to_idx(doc_id)
                iterations = [i // args.global_batch_size + 1 for i in indexs]
                iterations.sort()
                print("Iterations:", iterations, file=fout)
            print("=" * 80, file=fout)
            print(tokenizer.detokenize(text.tolist()), file=fout)
            print("=" * 80, file=fout)
        fout.close()

    
    