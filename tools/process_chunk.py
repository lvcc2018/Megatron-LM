import hashlib
import sys
import os
import time
from queue import Queue
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.data.dataset_utils import get_train_valid_test_split_
from megatron.tokenizer.tokenizer import _GPTSentencePieceTokenizer
import numpy as np
from tqdm import tqdm
import math


# DLM-3
# zh_split_token_id = [13, 30267, 30882, 30584] # \n, 。，？， ！
# en_split_token_id = [13, 29889, 29973, 29991, 1213] # \n, ., ?, !, ."
# code_split_token_id = [13]

# DLM-S
zh_split_token_id = [28, 59409, 59690, 59707] # \n, 。，？， ！
en_split_token_id = [28, 59401, 59767, 59807, 757] # \n, ., ?, !, ."
code_split_token_id = [28, 2] # \n, <eos>
split_token_id_dict = {
    'zh': zh_split_token_id,
    'en': en_split_token_id,
    'code': code_split_token_id
}

# split_token_id = code_split_token_id
# drop_long_document = False
# data_prefix = "/mnt/data01/shenyan/data/DLM-2-data/7B/zh45_en30_code6/zh45/zh45_text_document"
# data_prefix = "/mnt/data01/shenyan/data/DLM-2-data/7B/zh45_en30_code6/en30/en30_text_document"
# data_prefix = "/mnt/data01/shenyan/data/DLM-2-data/7B/7B/code_168G/code_168G_text_document"
# name="train"
# data_impl = "infer"
# max_seq_length = 4096
# batch_size = 1024
# num_samples = 10240
# # num_samples = 11028512
# full_document_ratio = 0.5
# buffer_size = 500
# tokenizer = _GPTSentencePieceTokenizer("/mnt/data01/shenyan/lvcc/DLM-2/Exp_7B/tokenizer/SentencePieceTokenizer/chinese_llama.model")
# pad_id = tokenizer._pad_id
# seed = 1403
# buffer = []
# splits_string = "998,1,1"

def init_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="data_path for megatron indexed dataset")
    parser.add_argument("--data_name", type=str, choices=['zh', 'en', 'code'], help="for data split rule")
    parser.add_argument("--split_name", type=str, choices=['train', 'valid', 'test'])
    parser.add_argument("--splits_string", type=str, default="998,1,1")
    parser.add_argument("--data_impl", type=str, default="infer")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--no-seperate-last-epoch", action="store_false", dest="seperate_last_epoch")
    parser.add_argument("--full_doc_ratio", type=float, default=0.5, help="Ratio of full document samples")
    parser.add_argument("--buffer_size", type=int, default=500)
    parser.add_argument("--tokenizer_model", type=str)
    parser.add_argument("--seed", type=int, default=1403)
    parser.add_argument("--no_shuffle_samples", action="store_false", help="debug", dest="shuffle_samples")
    parser.add_argument("--drop_long_doc", action="store_true")
    parser.add_argument("--only_check_length", action="store_true")
    parser.add_argument("--data_cache_dir", type=str, default=None)
    parser.add_argument("--multiprocess_chunksize", type=int, default=1000)
    parser.add_argument("--multiprocess_num", type=int, default=4)
    return parser

parser = init_parser()
args = parser.parse_args()

# init varibale from arguments
split_token_id = split_token_id_dict[args.data_name]
drop_long_document = args.drop_long_doc
data_prefix = args.data_path
name = args.split_name
data_impl = args.data_impl
max_seq_length = args.max_seq_length
batch_size = args.batch_size
num_samples = args.num_samples
# num_samples = 11028512
full_document_ratio = args.full_doc_ratio
buffer_size = args.buffer_size
tokenizer = _GPTSentencePieceTokenizer(args.tokenizer_model)
pad_id = tokenizer._pad_id
seed = args.seed
buffer = []
splits_string = args.splits_string
shuffle_samples = args.shuffle_samples
data_cache_dir = args.data_cache_dir
shuffle_samples = args.shuffle_samples
if data_cache_dir is None:
    data_cache_dir = os.path.join(os.path.dirname(data_prefix), 'index-cache')
    

class Sample(object):
    def __init__(self, is_long=False):
        self.doc_idx = []
        self.start_idx = []
        self.end_idx = []
        self.length = 0
        self.is_long = is_long
    
    def add(self, doc_idx, start_idx, end_idx):
        if doc_idx is not None:
            self.doc_idx.append(doc_idx)
        self.start_idx.append(start_idx)
        self.end_idx.append(end_idx)
        self.length += (end_idx - start_idx + 1)
    
    def __len__(self):
        return self.length
        
def init_batch(batch_size):
    batch_samples = []
    for i in range(batch_size):
        batch_samples.append(Sample())
    return batch_samples

def get_chunk_info_from_batch(batch_samples, pre_doc_idx):
    doc_idx = []
    sample_idx = []
    for sample in batch_samples:
        assert sample.length <= max_seq_length
        idx = len(doc_idx) + pre_doc_idx
        doc_idx.extend(sample.doc_idx)
        sample_idx.append([idx, sample.start_idx[0]])
    return doc_idx, sample_idx

def find_best_fit(batch_samples, max_seq_length, c_doc_size):
    # find best fit
    best_fit_length = -1
    best_fit_index = -1
    for j in range(len(batch_samples)):
        if batch_samples[j].is_long:
            # long sample can't be append
            continue
        left_length = max_seq_length - batch_samples[j].length
        if left_length >= c_doc_size and left_length > best_fit_length:
            best_fit_length = left_length
            best_fit_index = j
    return best_fit_index

def _build_shuffle_idx(num_samples, total_size, np_rng):
    """Build the range [0, size) and shuffle."""
    print(' > building shuffle index with split [0, {}) and [{}, {}) '
          '...'.format(num_samples, num_samples, total_size), flush=True)

    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples,
                                  step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size,
                                 step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))

def check_length(indexed_dataset, split_token_id):
    sizes = indexed_dataset.sizes
    exceed_num = 0
    split_exceed_num = 0
    exceed_indexes = []
    for index, size in enumerate(tqdm(sizes, "Check length", smoothing=0)):
        if size > max_seq_length:
            exceed_num += 1
            ids = indexed_dataset.get(index, 0)
            splited_ids = np.split(ids, np.where(np.isin(ids, split_token_id))[0] + 1)
            for ids in splited_ids:
                size = ids.shape[0]
                if size > max_seq_length:
                    split_exceed_num += 1
                    exceed_indexes.append(index)
                    break
    print("exceed max sequence length ratio: {} number: {}".format(exceed_num / sizes.shape[0], exceed_num))
    print("After split: exceed max sequence length ratio: {} number: {}".format(split_exceed_num / sizes.shape[0], split_exceed_num))
    np_rng = np.random.RandomState(seed)
    np_rng.shuffle(exceed_indexes)
    print("After split: exceed indexes", exceed_indexes[:100])

def split_document_to_samples(doc_idx, ids, split_token_id):
    if isinstance(split_token_id, int):
        split_token_id = [split_token_id]
    splited_ids = np.split(ids, np.where(np.isin(ids, split_token_id))[0] + 1)
    final_ids = []
    current_ids = np.array([], dtype=np.int32)
    current_samples = []
    offset = 0
    exceed_max_length = False
    for ids in splited_ids:
        size = ids.shape[0]
        if size > max_seq_length:
            # after split, the document is still longer than max_seq_length
            exceed_max_length = True
            break
    if exceed_max_length:
        if drop_long_document:
            return None, False, exceed_max_length
        else:
            ids = np.concatenate(splited_ids)
            size = ids.shape[0]
            num_chunk = int(size / max_seq_length)
            for i in range(num_chunk):
                sample = Sample(is_long=True)
                sample.add(None, offset, offset + max_seq_length - 1)
                current_samples.append(sample)
                offset += max_seq_length
            if offset < size:
                sample = Sample(is_long=False)
                sample.add(None, offset, size - 1)
                current_samples.append(sample)
            current_samples[-1].doc_idx.append(doc_idx)
            return current_samples, True, exceed_max_length
    else:
        for ids in splited_ids:
            size = ids.shape[0]
            if size > max_seq_length:
                # after split, the document is still longer than max_seq_length
                return None, False
            if current_ids.shape[0] + size > max_seq_length:
                final_ids.append(current_ids)
                sample = Sample(is_long=True)
                # print(current_ids.shape[0])
                sample.add(None, offset, offset + current_ids.shape[0] - 1)
                current_samples.append(sample)
                offset += current_ids.shape[0]
                current_ids = np.array([], dtype=np.int32)
            current_ids = np.concatenate((current_ids, ids))
        if current_ids.shape[0] > 0:
            final_ids.append(current_ids)
            sample = Sample(is_long=False)
            sample.add(None, offset, offset + current_ids.shape[0] - 1)
            current_samples.append(sample)
        # quit()
        current_samples[-1].doc_idx.append(doc_idx)
        return current_samples, True, exceed_max_length

def see_document(indexed_dataset, doc_idxs, tokenizer):
    for doc_idx in doc_idxs:
        ids = indexed_dataset.get(doc_idx, 0)
        print("Index:", doc_idx, "Length:", ids.shape[0])
        print(tokenizer.detokenize(ids.tolist()))
        print("=" * 80)
    
def build_normal_chunk(num_samples, indexed_dataset, shuffle_doc_idx, pre_doc_idx):
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)
    sizes = indexed_dataset.sizes
    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Begining offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = pre_doc_idx
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    pbar = tqdm(total=num_samples, desc="Normal Chunk", smoothing=0)
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = max_seq_length + 1
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = shuffle_doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += (remaining_seq_length + doc_length - 1)
                remaining_seq_length = 0
            else:
                # Otherwise, start from the begining of the next document.
                doc_idx_index += 1
                pre_doc_idx += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = pre_doc_idx
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1
        pbar.update(1)
    pbar.close()
    doc_idx = shuffle_doc_idx[:doc_idx_index + 1]
    return doc_idx, sample_idx, doc_idx_index

def find_unused_documents(doc_idx, shuffle_doc_idx, remove_documents):
    from collections import Counter
    doc_idx_counter = Counter(shuffle_doc_idx)
    for doc_id in doc_idx:
        doc_idx_counter[doc_id] -= 1
    for doc_id in remove_documents:
        doc_idx_counter[doc_id] -= 1
    unused_doc_idx = [doc_id for doc_id, count in doc_idx_counter.items() if count > 0]
    return np.array(unused_doc_idx, dtype=np.int32)

class Builder(object):
    def __init__(self):
        pass
    
    def initilize(self, indexed_dataset, shuffle_doc_idx):
        Builder.buffer = Queue(maxsize=buffer_size)
        Builder.batch_samples = []
        Builder.indexed_dataset = indexed_dataset
        Builder.shuffle_doc_idx = shuffle_doc_idx
        Builder.sizes = indexed_dataset.sizes
        Builder.remove_document_num = 0
        Builder.remove_documents = []
        Builder.long_sample_idxs = []
        Builder.exceed_sample_idxs = []
        
    
    def build_samples_from_doc(self, doc_idx_index):
        # print(doc_idx_index, len(Builder.batch_samples), Builder.buffer.qsize())
        clear_buffer = (len(Builder.batch_samples) == 0)
        if len(Builder.batch_samples) == 0:
            # init batch
            Builder.batch_samples = init_batch(batch_size)
        # process current document
        c_doc_idx = shuffle_doc_idx[doc_idx_index]
        c_doc_size = Builder.sizes[c_doc_idx]
        if c_doc_size > max_seq_length:
            # process long document
            new_samples, success, exceed_seq_length = split_document_to_samples(c_doc_idx, Builder.indexed_dataset.get(c_doc_idx, 0), split_token_id)
            if success:
                # add to current batch
                if not shuffle_samples:
                    for new_idx, sample in enumerate(new_samples):
                        if sample.is_long:
                            Builder.long_sample_idxs.append(len(Builder.batch_samples) + new_idx)
                        if exceed_seq_length:
                            Builder.exceed_sample_idxs.append(len(Builder.batch_samples) + new_idx)
                Builder.batch_samples.extend(new_samples)
            else:
                Builder.remove_document_num += 1
                Builder.remove_documents.append(c_doc_idx)
        else:
            # find best fit
            best_fit_index = find_best_fit(Builder.batch_samples, max_seq_length, c_doc_size)
            if best_fit_index != -1:
                Builder.batch_samples[best_fit_index].add(c_doc_idx, 0, c_doc_size - 1)
            else:
                # if buffer is not full, add to it
                if not Builder.buffer.full():
                    # Builder.buffer.append((c_doc_idx, 0, c_doc_size - 1))
                    Builder.buffer.put((c_doc_idx, 0, c_doc_size - 1))
                else:
                    # else save the chunk
                    batch_doc_idx, batch_sample_idx = get_chunk_info_from_batch(Builder.batch_samples, 0)
                    pad_number = 0
                    for sample in Builder.batch_samples:
                        pad_number += (max_seq_length - sample.length)
                    # clear batch
                    return_dict = {
                        'doc_idx': batch_doc_idx,
                        'sample_idx': batch_sample_idx,
                        'pad_number': pad_number,
                        'remove_doc_number': Builder.remove_document_num,
                        'remove_documents': Builder.remove_documents,
                        'doc_idx_index': doc_idx_index,
                        'long_sample_idxs': Builder.long_sample_idxs,
                        'exceed_sample_idxs': Builder.exceed_sample_idxs
                    }
                    Builder.batch_samples = []
                    Builder.long_sample_idxs = []
                    Builder.exceed_sample_idxs = []
                    Builder.remove_document_num = 0
                    Builder.remove_documents = []
                    result = self.build_samples_from_doc(doc_idx_index)
                    assert result is None
                    return return_dict
        if clear_buffer:
            # process buffer data and clear buffer
            qsize = Builder.buffer.qsize()
            for _ in range(qsize):
                (c_doc_idx, c_start_idx, c_end_idx) = Builder.buffer.get()
                c_doc_size = c_end_idx - c_start_idx + 1
                best_fit_index = find_best_fit(Builder.batch_samples, max_seq_length, c_doc_size)
                if best_fit_index != -1:
                    Builder.batch_samples[best_fit_index].add(c_doc_idx, c_start_idx, c_end_idx)
                else:
                    # no room for this doc, add to buffer
                    Builder.buffer.put((c_doc_idx, c_start_idx, c_end_idx))
        return None


if __name__ == "__main__":
    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix,
                                           data_impl,
                                           skip_warmup=True)
    log_str = ""
    print(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))
    print('    number of documents: {}'.format(
        indexed_dataset.sizes.shape[0]))
    log_str += ' > finished creating indexed dataset in {:4f} seconds\n'.format(time.time() - start_time)
    log_str += '    number of documents: {}\n'.format(indexed_dataset.sizes.shape[0])
    if args.only_check_length:
        check_length(indexed_dataset, split_token_id)
        quit()
    # see_document(indexed_dataset, [10102625, 12190992, 10523881, 9644792, 8287225, 10483214, 11697000, 14037879, 10070101, 7887919], tokenizer)
    # quit()
    sizes = indexed_dataset.sizes
    splits = get_train_valid_test_split_(splits_string, sizes.shape[0])
    if name == 'train':
        start = splits[0]
        stop = splits[1]
    elif name == 'valid':
        start = splits[1]
        stop = splits[2]
    elif name == 'test':
        start = splits[2]
        stop = splits[3]
    else:
        raise ValueError('invalid dataset split: {}'.format(name))
    print(f"{name} documents is [{start}, {stop})")
    log_str += f"{name} documents is [{start}, {stop})\n"
    shuffle_doc_idxs = []
    for epoch in range(args.epochs - 1):
        shuffle_doc_idx = np.arange(start=start, stop=stop, step=1, dtype=np.int32)
        shuffle_doc_idxs.append(shuffle_doc_idx)
    if args.seperate_last_epoch:
        shuffle_doc_idx_last = np.arange(start=start, stop=stop, step=1, dtype=np.int32)
    else:
        shuffle_doc_idx_last = np.arange(start=start, stop=stop, step=1, dtype=np.int32)
        shuffle_doc_idxs.append(shuffle_doc_idx_last)
    shuffle_doc_idx = np.concatenate(shuffle_doc_idxs)
    sample_idx = []
    np_rng = np.random.RandomState(seed)
    np_rng.shuffle(shuffle_doc_idx)
    if args.seperate_last_epoch:
        np_rng.shuffle(shuffle_doc_idx_last)
        shuffle_doc_idx = np.concatenate((shuffle_doc_idx, shuffle_doc_idx_last))
    doc_idx = []
    processed_samples = 0
    processed_doc_num = 0
    pad_number = 0
    long_sample_idxs = []
    exceed_sample_idxs = []
    remove_document_num = 0
    remove_documents = []
    full_document_num = math.ceil(num_samples * full_document_ratio)
    start_time = time.time()
    if args.multiprocess_num > 0:
        print(f"Process with {args.multiprocess_num} processes ...")
        log_str += f"Process with {args.multiprocess_num} processes ...\n"
        from multiprocessing import Pool, Manager
        builder = Builder()
        pools = Pool(args.multiprocess_num, initializer=builder.initilize, initargs=(indexed_dataset, shuffle_doc_idx))
        pbar = tqdm(total=full_document_num, desc="Full Document", smoothing=0)
        for result in pools.imap(builder.build_samples_from_doc, range(shuffle_doc_idx.shape[0]), chunksize=args.multiprocess_chunksize):
            # print(processed_doc_num_2, len(sample_idx))
            if result is None:
                continue
            batch_doc_idx = result['doc_idx']
            batch_sample_idx = result['sample_idx']
            for i in range(len(batch_sample_idx)):
                batch_sample_idx[i][0] = len(doc_idx) + batch_sample_idx[i][0]
            if not shuffle_samples:
                long_sample_idxs.extend([i + len(doc_idx) for i in result['long_sample_idxs']])
                exceed_sample_idxs.extend([i + len(doc_idx) for i in result['exceed_sample_idxs']])
            pbar.update(min(len(batch_sample_idx), full_document_num - len(sample_idx)))
            doc_idx.extend(batch_doc_idx)
            sample_idx.extend(batch_sample_idx)
            pad_number += result['pad_number']
            remove_document_num += result['remove_doc_number']
            remove_documents.extend(result['remove_documents'])
            processed_doc_num = result['doc_idx_index']
            if len(sample_idx) >= full_document_num:
                break
        pbar.close()
        pools.close()
    else:
        print(f"Process with single process ...")
        log_str += f"Process with single process ...\n"
        builder = Builder()
        builder.initilize(indexed_dataset, shuffle_doc_idx)
        pbar = tqdm(total=full_document_num, desc="Full Document", smoothing=0)
        for i in range(shuffle_doc_idx.shape[0]):
            result = builder.build_samples_from_doc(i)
            if result is None:
                continue
            batch_doc_idx = result['doc_idx']
            batch_sample_idx = result['sample_idx']
            for i in range(len(batch_sample_idx)):
                batch_sample_idx[i][0] = len(doc_idx) + batch_sample_idx[i][0]
            if not shuffle_samples:
                long_sample_idxs.extend([i + len(doc_idx) for i in result['long_sample_idxs']])
                exceed_sample_idxs.extend([i + len(doc_idx) for i in result['exceed_sample_idxs']])
            pbar.update(min(len(batch_sample_idx), full_document_num - len(sample_idx)))
            doc_idx.extend(batch_doc_idx)
            sample_idx.extend(batch_sample_idx)
            pad_number += result['pad_number']
            remove_document_num += result['remove_doc_number']
            remove_documents.extend(result['remove_documents'])
            processed_doc_num = result['doc_idx_index']
            if len(sample_idx) >= full_document_num:
                break
        pbar.close()
    # postprocess to add last sample idx or build normal chunk
    print(f"Build {len(sample_idx)} full document samples from {len(doc_idx)} documents in {time.time() - start_time} seconds")
    log_str += f"Build {len(sample_idx)} full document samples from {len(doc_idx)} documents in {time.time() - start_time} seconds\n"
    if full_document_ratio == 1.0:
        # add last sample idx
        full_document_num = len(sample_idx)
        if buffer.empty():
            sample_idx.extend([len(doc_idx), 0])
            doc_idx.append(processed_doc_num)
        else:
            c_doc_idx, c_start_idx, c_end_idx = buffer.get()
            sample_idx.append([len(doc_idx), c_start_idx])
            doc_idx.append(c_doc_idx)
        sample_idx = np.array(sample_idx, dtype=np.int32)
        doc_idx = np.array(doc_idx, dtype=np.int32)
    else:
        # add normal chunk
        # find unused doc idx
        unused_doc_idx = find_unused_documents(doc_idx, shuffle_doc_idx[:processed_doc_num].tolist(), remove_documents)
        print(f"Building normal chunk after {processed_doc_num} documents besides {len(unused_doc_idx)} unused documents before ...")
        log_str += f"Building normal chunk after {processed_doc_num} documents besides {len(unused_doc_idx)} unused documents before ...\n"
        full_document_num = len(sample_idx)
        normal_chunk_num = int((1 - full_document_ratio) / full_document_ratio * full_document_num)
        shuffle_doc_idx = np.concatenate((unused_doc_idx, shuffle_doc_idx[processed_doc_num:]))
        normal_doc_idx, normal_sample_idx, normal_chunk_processed_doc_num = build_normal_chunk(normal_chunk_num, indexed_dataset, shuffle_doc_idx, len(doc_idx))
        print(f"Build {len(normal_sample_idx) - 1} from {normal_chunk_processed_doc_num} documents")
        log_str += f"Build {len(normal_sample_idx) - 1} from {normal_chunk_processed_doc_num} documents\n"
        sample_idx = np.array(sample_idx, dtype=np.int32)
        doc_idx = np.array(doc_idx, dtype=np.int32)
        doc_idx = np.concatenate((doc_idx, normal_doc_idx))
        sample_idx = np.concatenate((sample_idx, normal_sample_idx))
    processed_doc_num = doc_idx.shape[0] + len(remove_documents)
    print(f"build doc_idx and sample_idx in {time.time() - start_time} seconds, build {sample_idx.shape[0] - 1} samples")
    print(f"process epoch: {processed_doc_num / (stop - start)}")
    print(f"remove ratio: {remove_document_num} / {processed_doc_num} = {remove_document_num / processed_doc_num}")
    print("pad ratio: {}".format(pad_number / (full_document_num * max_seq_length)))
    log_str += f"build doc_idx and sample_idx in {time.time() - start_time} seconds, build {sample_idx.shape[0] - 1} samples\n"
    log_str += f"process epoch: {processed_doc_num / (stop - start)}\n"
    log_str += f"remove ratio: {remove_document_num} / {processed_doc_num} = {remove_document_num / processed_doc_num}\n"
    log_str += "pad ratio: {}\n".format(pad_number / (full_document_num * max_seq_length))
    if args.shuffle_samples:
        shuffle_idx = _build_shuffle_idx(sample_idx.shape[0] - 1, sample_idx.shape[0] - 1, np_rng)
    else:
        shuffle_idx = np.arange(start=0, stop=sample_idx.shape[0], step=1, dtype=np.uint32)
    # Filename of the index mappings.
    desc = "GPT Dataset\n\n"
    desc += f"Data prefix {data_prefix}\n"
    desc += f"Dataset name {name}\n"
    desc += f"Number of samples {num_samples}\n"
    desc += f"Sequence length {max_seq_length}\n"
    desc += f"Random seed {seed}\n"
    desc += f"Split {splits_string}\n"
    desc_hash = hashlib.md5(desc.encode('utf-8')).hexdigest()
    desc_filename = desc_hash + ".dsc"
    doc_idx_filename = desc_hash + '_doc_idx.npy'
    sample_idx_filename = desc_hash + '_sample_idx.npy'
    shuffle_idx_filename = desc_hash + '_shuffle_idx.npy'
    log_filename = desc_hash + '.log'
    desc_filename = os.path.join(data_cache_dir, desc_filename)
    doc_idx_filename = os.path.join(data_cache_dir, doc_idx_filename)
    sample_idx_filename = os.path.join(data_cache_dir, sample_idx_filename)
    shuffle_idx_filename = os.path.join(data_cache_dir, shuffle_idx_filename)
    log_filename = os.path.join(data_cache_dir, log_filename)
    os.makedirs(data_cache_dir, exist_ok=True)
    # description
    with open(desc_filename, 'wt') as fd:
        fd.write(desc)
    np.save(doc_idx_filename, doc_idx, allow_pickle=True)
    np.save(sample_idx_filename, sample_idx, allow_pickle=True)
    np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
    print("Save doc_idx, sample_idx, shuffle_idx to {}".format(os.path.join(data_cache_dir, desc_hash)))
    log_str += "Save doc_idx, sample_idx, shuffle_idx to {}\n".format(os.path.join(data_cache_dir, desc_hash))
    if not args.shuffle_samples:
        print("Exceed sequence length idxs:", exceed_sample_idxs)
        log_str += "Exceed sequence length idxs: {}\n".format(exceed_sample_idxs)
        print("Long sample idxs:", long_sample_idxs)
        log_str += "Long sample idxs: {}\n".format(long_sample_idxs)
    with open(log_filename, 'wt') as fd:
        fd.write(log_str)
    
    
    
    
