import argparse
import sys
import numpy as np
import h5py
from tqdm import tqdm


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--megatron-path', type=str, default=None,)
    parser.add_argument('--data-paths', nargs='+', default=None,)
    parser.add_argument('--output-path', type=str, default=None,)
    parser.add_argument('--max-seq-length', type=int, default=2048)
    parser.add_argument('--tokenizer-type', type=str, default=None,
                       choices=['MixedTokenizer',
                                'GPTSentencePieceTokenizer'],
                       help='What type of tokenizer to use.')
    parser.add_argument('--tokenizer-model', type=str, default=None,
                       help='Sentencepiece tokenizer model.')
    parser.add_argument('--tokenizer-file', type=str, default=None,
                       help='BPE tokenizer file.')
    parser.add_argument('--use-mix-format', action='store_true', default=None)
    parser.add_argument('--use-cache', action='store_true', default=None)
    parser.add_argument('--seed', type=int, default=42, help='random seed for data shuffle')
    return parser

def shuffle_and_pack_data(samples, seed, max_seq_length, pad_token_id, tokenizer):
    np_rng = np.random.RandomState(seed)
    np_rng.shuffle(samples)
    i = 0
    def reset_packed_sample():
        return  {
                'input_ids': [],
                'labels': [],
                'loss_mask': [],
                'segment_ids': [],
            }
    packed_sample = reset_packed_sample()
    segment_id = 0
    packed_samples = []
    while i < len(samples):
        current_sample = samples[i]
        input_ids = current_sample['input_ids']
        labels = current_sample['labels']
        loss_mask = current_sample['loss_mask']
        if len(packed_sample['input_ids']) + len(input_ids) <= max_seq_length:
            packed_sample['input_ids'].extend(input_ids)
            packed_sample['labels'].extend(labels)
            packed_sample['loss_mask'].extend(loss_mask)
            packed_sample['segment_ids'].extend([segment_id] * len(input_ids))
            segment_id += 1
            i += 1
        else:
            left_number = max_seq_length - len(packed_sample['input_ids'])
            packed_sample['input_ids'].extend([pad_token_id] * left_number)
            packed_sample['labels'].extend([pad_token_id] * left_number)
            packed_sample['loss_mask'].extend([0] * left_number)
            packed_sample['segment_ids'].extend([segment_id] * left_number)
            packed_sample['num_sample'] = segment_id
            packed_samples.append(packed_sample)
            packed_sample = reset_packed_sample()
            segment_id = 0
    print(f"Pack {len(samples)} to {len(packed_samples)} samples")
    # for s in packed_samples:
    #     print(f"  >> num_sample: {s['num_sample']}")
    #     segment_ids = s['segment_ids']
    #     next_segment_id = 1
    #     start_index = 0
    #     while next_segment_id < s['num_sample']:
    #         end_index = segment_ids.index(next_segment_id)
    #         print(f"   > sample {next_segment_id - 1}:")
    #         print(tokenizer.detokenize(s['input_ids'][start_index:end_index]))
    #         start_index = end_index
    #         next_segment_id += 1
    #     quit()
        
    return packed_samples

def write_hdf5(samples, output_path, max_seq_length, use_mix_format):
    f = h5py.File(output_path, 'w')
    input_ids = f.create_dataset("input_ids", (len(samples), max_seq_length), dtype='i')
    labels = f.create_dataset("labels", (len(samples), max_seq_length), dtype='i')
    loss_mask = f.create_dataset("loss_mask", (len(samples), max_seq_length), dtype='i')
    segment_ids = f.create_dataset("segment_ids", (len(samples), max_seq_length), dtype='i')
    num_sample = f.create_dataset("num_sample", (len(samples,)), dtype='i')
    for i, s in enumerate(tqdm(samples, desc="Writing to hdf5")):
        input_ids[i, :] = s['input_ids']
        labels[i, :] = s['labels']
        loss_mask[i, :] = s['loss_mask']
        segment_ids[i, :] = s['segment_ids']
        num_sample[i] = s['num_sample']
    f['use_mix_format'] = use_mix_format
    f['seq_length'] = max_seq_length
    f['samples_number'] = len(samples)
    f.close()
    print(f"Write {len(samples)} samples to {output_path}")

def read_hdf5(path, tokenizer):
    f = h5py.File(path, 'r')
    input_ids = f['input_ids']
    labels = f['labels']
    loss_mask = f['loss_mask']
    segment_ids = f['segment_ids']
    num_sample = f['num_sample']
    samples_number = f['samples_number'][()]
    use_mix_format = f['use_mix_format'][()]
    max_seq_length = f['seq_length'][()]
    print(samples_number)
    print(use_mix_format)
    print(max_seq_length)
    quit()
    for i in range(1):
        print(f"  >> num_sample: {num_sample[i]}")
        c_segment_ids = segment_ids[i].tolist()
        next_segment_id = 1
        start_index = 0
        c_input_ids = input_ids[i].tolist()
        while next_segment_id <= num_sample[i]:
            end_index = c_segment_ids.index(next_segment_id)
            print(f"   > sample {next_segment_id - 1}:")
            print(tokenizer.detokenize(c_input_ids[start_index:end_index]))
            start_index = end_index
            next_segment_id += 1
        quit()
    
if __name__=="__main__":
    parser = init_parser()
    args = parser.parse_args()
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    from data import process_single_datapath, PAD_TOKEN_ID
    from megatron.tokenizer.tokenizer import _SentencePieceTokenizer, _MixedTokenizer, _GPTSentencePieceTokenizer
    data_paths = args.data_paths
    
    if args.tokenizer_type == 'MixedTokenizer':
        tokenizer = _MixedTokenizer(args.tokenizer_model, args.tokenizer_file)
    elif args.tokenizer_type == 'GPTSentencePieceTokenizer':
        tokenizer = _GPTSentencePieceTokenizer(args.tokenizer_model)
    else:
        raise ValueError('Unknown tokenizer type: {}'.format(args.tokenizer_type))
    samples = []
    for path in data_paths:
        current_samples = process_single_datapath(path, tokenizer, args.max_seq_length, args.use_mix_format, args.use_cache)
        samples.extend(current_samples)
    print('  >> total number of samples: {}'.format(len(samples)))
    packed_samples = shuffle_and_pack_data(samples, args.seed, args.max_seq_length, PAD_TOKEN_ID, tokenizer)
    write_hdf5(packed_samples, args.output_path, args.max_seq_length, args.use_mix_format)
    # read_hdf5(args.output_path, tokenizer)
    
    
    