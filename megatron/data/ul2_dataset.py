# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""GPT style dataset."""

import os
import time

import numpy as np
import math
import torch

from megatron import print_rank_0, get_tokenizer
from megatron.core import mpu
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples_withname
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples_for_dataset_manager
from megatron.data.dataset_utils import get_train_valid_test_split_
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.data.dataset_manager import DatasetManager, WeightScheduler


def build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                    train_valid_test_num_samples,
                                    seq_length, seed, skip_warmup,
                                    train_data_prefix=None,
                                    valid_data_prefix=None,
                                    test_data_prefix=None,
                                    return_doc_ids=False,
                                    use_dataloader_manager=False,
                                    use_dataset_manager=False,
                                    global_batch_size=None):
    """Build train, valid, and test datasets."""
    if use_dataset_manager:
        train_valid_test_num_iters = [num_samples // global_batch_size for num_samples in train_valid_test_num_samples]

    if data_prefix:
        print_rank_0("Single data path provided for train, valid & test")

        # Single dataset.
        if len(data_prefix) == 1:
            return _build_train_valid_test_datasets(data_prefix[0],
                                                    data_impl, splits_string,
                                                    train_valid_test_num_samples,
                                                    seq_length, seed, skip_warmup)

        # Blending dataset.
        # Parse the values.
        if use_dataloader_manager:
            output = get_datasets_weights_and_num_samples_withname(data_prefix,
                                                    train_valid_test_num_samples)
            prefixes, weights, names, datasets_train_valid_test_num_samples = output
        elif use_dataset_manager:
            output = get_datasets_weights_and_num_samples_for_dataset_manager(data_prefix,
                                                    train_valid_test_num_samples, global_batch_size, 
                                                    train_valid_test_num_iters)
            prefixes, weight_schedulers, names, datasets_train_valid_test_num_samples = output
        else:
            output = get_datasets_weights_and_num_samples(data_prefix,
                                                        train_valid_test_num_samples)
            prefixes, weights, datasets_train_valid_test_num_samples = output
        
        train_num_samples, valid_num_samples, test_num_samples = map(
            sum,
            zip(*datasets_train_valid_test_num_samples)
        )

        # Build individual datasets.
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        for i in range(len(prefixes)):
            train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
                prefixes[i], data_impl, splits_string,
                datasets_train_valid_test_num_samples[i],
                seq_length, seed, skip_warmup,
                return_doc_ids)
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)
        
        if use_dataloader_manager:
            # We will use DataLoaderManager, so no need to blend here.
            return ((train_datasets, weights, names), (valid_datasets, weights, names), (test_datasets, weights, names))
        
        if use_dataset_manager:
            # We will use DatasetManager, so no need to blend here.
            train_dataset = DatasetManager(names, train_datasets, weight_schedulers[0], global_batch_size, train_valid_test_num_iters[0])
            valid_dataset = DatasetManager(names, valid_datasets, weight_schedulers[1], global_batch_size, train_valid_test_num_iters[1])
            test_dataset = DatasetManager(names, test_datasets, weight_schedulers[2], global_batch_size, train_valid_test_num_iters[2])
            return (train_dataset, valid_dataset, test_dataset)

        # Blend.
        blending_train_dataset = None
        if train_datasets:
            blending_train_dataset = BlendableDataset(train_datasets, weights, train_num_samples)
        blending_valid_dataset = None
        if valid_datasets:
            blending_valid_dataset = BlendableDataset(valid_datasets, weights, valid_num_samples)
        blending_test_dataset = None
        if test_datasets:
            blending_test_dataset = BlendableDataset(test_datasets, weights, test_num_samples)

        return (blending_train_dataset, blending_valid_dataset,
                blending_test_dataset)

    else:
        print_rank_0("Separate data paths provided for train, valid & test. Split string will be ignored.")

        train_dataset, valid_dataset, test_dataset = None, None, None
        # Single dataset.
        if train_data_prefix is not None:
            train_dataset = build_dataset("train", train_data_prefix, data_impl,
                                          train_valid_test_num_samples[0],
                                          seq_length, seed, skip_warmup)

        if valid_data_prefix is not None:
            valid_dataset = build_dataset("valid", valid_data_prefix, data_impl,
                                          train_valid_test_num_samples[1],
                                          seq_length, seed, False)

        if test_data_prefix is not None:
            test_dataset = build_dataset("test", test_data_prefix, data_impl,
                                         train_valid_test_num_samples[2],
                                         seq_length, seed, False)

        return (train_dataset, valid_dataset, test_dataset)


def _build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                     train_valid_test_num_samples,
                                     seq_length, seed, skip_warmup,
                                     return_doc_ids=False):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1],
                                  step=1, dtype=np.int32)
            dataset = UL2Dataset(name, data_prefix,
                                 documents, indexed_dataset,
                                 train_valid_test_num_samples[index],
                                 seq_length, seed,
                                 return_doc_ids)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


def build_dataset(dataset_name, data_prefix, data_impl, num_samples,
                  seq_length, seed, skip_warmup):
    dataset = None
    if len(data_prefix) == 1:
        dataset = _build_dataset(dataset_name,
                        data_prefix[0], data_impl,
                        num_samples, seq_length,
                        seed, skip_warmup)
    else:
        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        prefixes, weights, dataset_num_samples = output
        num_samples = sum(dataset_num_samples)

        # Build individual datasets.
        datasets = []
        for i in range(len(prefixes)):
            ds = _build_dataset(dataset_name, prefixes[i],
                            data_impl, dataset_num_samples[i],
                            seq_length, seed, skip_warmup)
            if ds:
                datasets.append(ds)

        if datasets:
            dataset = (datasets, weights)
            # We will use DataLoaderManager, so no need to blend here.
            # dataset = BlendableDataset(datasets, weights)

    return dataset


def _build_dataset(dataset_name, data_prefix, data_impl,
                   num_samples, seq_length, seed, skip_warmup):
    """
    Build dataset. This method is called when individual
    train, valid, test datasets are provided
    """

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]

    print_rank_0('    {}:'.format(dataset_name))
    print_rank_0('     document indices in [0, {}) total of {} '
                 'documents'.format(total_num_of_documents, total_num_of_documents))

    documents = np.arange(start=0, stop=total_num_of_documents,
                        step=1, dtype=np.int32)

    dataset = UL2Dataset(dataset_name, data_prefix,
                        documents, indexed_dataset,
                        num_samples, seq_length, seed)

    return dataset


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):
    """Build indexed dataset."""
    print_rank_0(' > building dataset index ...')

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix,
                                           data_impl,
                                           skip_warmup)
    print_rank_0(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))
    print_rank_0('    number of documents: {}'.format(
        indexed_dataset.sizes.shape[0]))

    return indexed_dataset


class UL2Dataset(torch.utils.data.Dataset):

    def __init__(self, name, data_prefix, documents, indexed_dataset, num_samples, seq_length, seed, return_doc_ids=False):
        self.name = name
        self.indexed_dataset = indexed_dataset
        self.return_doc_ids = return_doc_ids

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx, self.index_prefix, self.denoisers, self.denoisers_info, self.seq_length_per_sample = _build_index_mappings(
            self.name, data_prefix, documents, self.indexed_dataset.sizes, num_samples, seq_length, seed)
        self.seed = seed
        self.seq_length = seq_length
        self.tokenizer = get_tokenizer()
        self.special_tokens = self.tokenizer._special_tokens
        self.extra_id_begin = len(self.tokenizer._vocab) - len(self.special_tokens) + 6
        # 6 is for [X] [S] [R] <s> </s> <pad>

        # self.sample_rngs = [np.random.RandomState(seed + i + 1) for i in range(len(self))]

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        doc_ids = []
        if doc_index_f == doc_index_l:
            doc_ids.append(self.doc_idx[doc_index_f])
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            doc_ids.append(self.doc_idx[doc_index_f])
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                                    offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                doc_ids.append(self.doc_idx[i])
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            doc_ids.append(self.doc_idx[doc_index_l])
            sample_list.append(self.indexed_dataset.get(
                self.doc_idx[doc_index_l],
                length=offset_l + 1))
            sample = np.concatenate(sample_list)
        try:
            denoiser_idx = self.denoisers[idx]
        except IndexError:
            print(len(self.denoisers), idx)
            quit()
        denoiser_info = self.denoisers_info[denoiser_idx]
        np_rng = np.random.RandomState(seed=self.seed + idx)
        total_inputs, attention_mask, loss_mask, position_ids = build_training_sample(sample, denoiser_idx, denoiser_info, np_rng, self.seq_length, self.extra_id_begin, self.tokenizer)
        return_dict = {
            'text': total_inputs,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
        }

        if self.return_doc_ids: # for retro preprocessing
            return_dict['doc_ids'] = np.array(doc_ids, dtype=np.int64)
            return return_dict
        else:
            return return_dict

def build_training_sample(tokens, denoiser, denoiser_info, np_rng, max_seq_length, extra_id_begin, tokenizer):
    from megatron import get_args
    args = get_args()
    pad_id = tokenizer._pad_id
    # if args.rank == 0:
    #     print(denoiser_info, denoiser)
    #     text = tokenizer.detokenize(tokens.tolist())
    #     text = text.replace('\n', '\\n')
    #     print(text)
    if denoiser == 6:
        # S Denoiser
        def prefix_lm_noise_mask_fn():
            token_length = len(tokens)
            max_input_length = (token_length - 1)
            min_input_length = token_length // 2
            split_index = np_rng.randint(min_input_length, max_input_length + 1)
            noise_mask = np.ones((token_length, ), dtype=np.bool_)
            noise_mask[:split_index] = False
            return noise_mask, split_index
        def prefix_lm_inputs_fn(noise_mask):
            new_tokens = np.array(tokens, dtype=np.int64)
            inputs = new_tokens[np.logical_not(noise_mask)]
            return inputs
        def prefix_lm_targets_fn(noise_mask):
            return prefix_lm_inputs_fn(np.logical_not(noise_mask))
        def prefix_lm_denoise():
            noise_mask, split_index = prefix_lm_noise_mask_fn()
            inputs = prefix_lm_inputs_fn(noise_mask)
            targets = prefix_lm_targets_fn(noise_mask)
            denoiser_type = denoiser_config[denoiser][2]
            denoiser_id = tokenizer._special_tokens[denoiser_type]
            total_inputs = np.concatenate([[denoiser_id], inputs, [pad_id], targets]).astype(np.int64)
            bi_length = inputs.shape[0] + 1 # 1 is for denoiser_id
            uni_length = max_seq_length - bi_length
            bi_attention_mask = np.zeros((max_seq_length, bi_length), dtype=np.int64)
            uni_attention_mask = np.triu(np.ones((uni_length, uni_length), dtype=np.int64), k=1)
            uni_attention_mask = np.concatenate(
                [np.ones((bi_length, uni_length), dtype=np.int64), uni_attention_mask],
                axis=0,
            )
            attention_mask = np.concatenate([bi_attention_mask, uni_attention_mask], axis=1)
            loss_mask = np.concatenate([np.zeros(bi_length, dtype=np.int64), np.ones(uni_length, dtype=np.int64)], axis=0)
            position_ids = np.arange(max_seq_length, dtype=np.int64)
            return total_inputs, attention_mask, loss_mask, position_ids
        total_inputs, attention_mask, loss_mask, position_ids = prefix_lm_denoise()
    else:
        # X, R Denoiser
        # print(denoiser_info)
        noise_span_number = denoiser_info[1]
        density = denoiser_config[denoiser][1]
        token_length = len(tokens)
        noise_token_number = round(token_length * density)
        nonoise_token_number = token_length - noise_token_number
        def _random_segmentation(num_items, num_segments):
            first_in_segment = (np.arange(num_items - 1) < (num_segments - 1)).astype(np.int32)
            np_rng.shuffle(first_in_segment)
            first_in_segment = np.concatenate([[0], first_in_segment])
            segment_id = np.cumsum(first_in_segment)
            sequence_lengths = []
            # TODO: more efficient method
            for i in range(num_segments):
                sequence_lengths.append(np.sum(segment_id == i))
            return sequence_lengths
        def noise_mask_fn():
            noise_span_lengths = _random_segmentation(noise_token_number, noise_span_number)
            nonoise_span_lengths = _random_segmentation(nonoise_token_number, noise_span_number)
            interleaved_span_lengths = np.stack([nonoise_span_lengths, noise_span_lengths], axis=1).reshape([-1])
            span_starts = np.cumsum(interleaved_span_lengths)[:-1]
            span_start_indicator = np.zeros(token_length, dtype=np.int32)
            span_start_indicator[span_starts] = 1
            span_num = np.cumsum(span_start_indicator)
            is_noise = (span_num % 2 == 1)
            return is_noise
        def inputs_fn(noise_mask):
            prev_token_is_noise = np.concatenate([[False], noise_mask[:-1]])
            first_noise_tokens = np.logical_and(noise_mask, np.logical_not(prev_token_is_noise))
            subsequent_noise_tokens = np.logical_and(noise_mask, prev_token_is_noise)
            new_tokens = np.array(tokens, dtype=np.int64)
            sentinel_ids = np.arange(noise_span_number).astype(np.int64) + extra_id_begin
            new_tokens[first_noise_tokens] = sentinel_ids
            input_tokens = new_tokens[np.logical_not(subsequent_noise_tokens)]
            return input_tokens
        def targets_fn(noise_mask):
            targets = inputs_fn(np.logical_not(noise_mask))
            # targets = np.concatenate([targets, [extra_id_begin + noise_span_number]])
            return targets
        def denoise():
            noise_mask = noise_mask_fn()
            inputs = inputs_fn(noise_mask)
            targets = targets_fn(noise_mask)
            denoiser_type = denoiser_config[denoiser][2]
            denoiser_id = tokenizer._special_tokens[denoiser_type]
            if denoiser_info[2] > 0: # add pad
                total_inputs = np.concatenate([[denoiser_id], inputs, [pad_id], targets, [pad_id] * denoiser_info[2]]).astype(np.int64)
            else:
                total_inputs = np.concatenate([[denoiser_id], inputs, [pad_id], targets]).astype(np.int64)
            bi_length = inputs.shape[0] + 1 # 1 is for denoiser_id
            uni_length = max_seq_length - bi_length
            bi_attention_mask = np.zeros((max_seq_length, bi_length), dtype=np.int64)
            uni_attention_mask = np.triu(np.ones((uni_length, uni_length), dtype=np.int64), k=1)
            uni_attention_mask = np.concatenate(
                [np.ones((bi_length, uni_length), dtype=np.int64), uni_attention_mask],
                axis=0,
            )
            attention_mask = np.concatenate([bi_attention_mask, uni_attention_mask], axis=1)
            loss_mask = np.concatenate([np.zeros(bi_length, dtype=np.int64), np.ones(uni_length, dtype=np.int64)], axis=0)
            position_ids = np.arange(max_seq_length, dtype=np.int64)
            return total_inputs, attention_mask, loss_mask, position_ids
        total_inputs, attention_mask, loss_mask, position_ids = denoise()
    if total_inputs.shape[0] < max_seq_length + 1:
        total_inputs = np.concatenate([total_inputs, [tokenizer._pad_id] * (max_seq_length + 1 - total_inputs.shape[0])])
    assert total_inputs.shape == (max_seq_length + 1,)
    assert attention_mask.shape == (max_seq_length, max_seq_length)
    assert loss_mask.shape == (max_seq_length,)
    assert position_ids.shape == (max_seq_length,)
    return total_inputs, attention_mask, loss_mask, position_ids
        


def _build_index_mappings(name, data_prefix, documents, sizes,
                          num_samples, seq_length, seed):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Calculate the actual length of each sample
    denoisers = _assign_denoiser(np_rng, num_samples)
    denoisers_info = _get_all_denoisers_info(seq_length)
    seq_length_per_sample = [denoisers_info[denoiser][0] for denoiser in denoisers]
    
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length_per_sample)

    # assert num_epochs <= 1, "some unpredictable errors may occur when num_epochs > 1."

    # Calculate the number of tokens that need to be supplemented in order
    # to correct the value of num_samples.
    original_total_tokens = sum(seq_length_per_sample)
    new_total_tokens = num_epochs * tokens_per_epoch
    remain_tokens = new_total_tokens - original_total_tokens

    # Filename of the index mappings.
    index_prefix = '{}_ul2_indexmap'.format(name)
    index_prefix += '_{}ns'.format(num_samples)
    index_prefix += '_{}sl'.format(seq_length)
    index_prefix += '_{}s'.format(seed)
    _filename = data_prefix + '_' + index_prefix
    doc_idx_filename = _filename + '_doc_idx.npy'
    sample_idx_filename = _filename + '_sample_idx.npy'
    shuffle_idx_filename = _filename + '_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0:
        if (not os.path.isfile(doc_idx_filename)) or \
           (not os.path.isfile(sample_idx_filename)) or \
           (not os.path.isfile(shuffle_idx_filename)):

            print_rank_0(' > WARNING: could not find index map files, building '
                         'the indices on rank 0 ...')

            # For the last epoch, decide whether include the entire epoch
            # in the global shuffle or not.

            # If we need only one epoch, then separating last epoch  does
            # not mean anything.
            # if num_epochs == 1:
            #     separate_last_epoch = False
            #     print(' > only one epoch required, setting '
            #           'separate_last_epoch to False', flush=True)

            # else:
            #     # Get the number of samples for the last epoch
            #     num_samples_from_epochs_minus_one = (
            #         (num_epochs - 1) * tokens_per_epoch - 1) // seq_length
            #     last_epoch_num_samples = num_samples - \
            #                              num_samples_from_epochs_minus_one
            #     assert last_epoch_num_samples >= 0, \
            #         'last epoch number of samples should be non-negative.'
            #     num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
            #     assert last_epoch_num_samples < (num_samples_per_epoch + 1), \
            #         'last epoch number of samples exceeded max value.'
            #     # If we have less than 80% of the samples for the last epoch,
            #     # seperate out the epoch and treat it differently.
            #     # Note: the 80% number is just based on common sense and can
            #     # be adjusted if needed.
            #     separate_last_epoch = (last_epoch_num_samples <
            #                            int(0.80 * num_samples_per_epoch))
            #     if separate_last_epoch:
            #         string = ' > last epoch number of samples ({}) is smaller '\
            #                  'than 80% of number of samples per epoch ({}), '\
            #                  'setting separate_last_epoch to True'
            #     else:
            #         string = ' > last epoch number of samples ({}) is larger '\
            #                  'than 80% of number of samples per epoch ({}), '\
            #                  'setting separate_last_epoch to False'
            #     print(string.format(last_epoch_num_samples,
            #                         num_samples_per_epoch), flush=True)

            separate_last_epoch = False

            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng,
                                     separate_last_epoch)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save doc-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            # First compile and then import.

            from megatron.data import helpers
            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32
            # sample_idx, denoisers, seq_length_per_sample = helpers.build_sample_idx_ul2(
            #     sizes, doc_idx, seq_length_per_sample, denoisers, seed, remain_tokens)
            sample_idx, denoisers, seq_length_per_sample = _build_sample_idx(
                sizes, doc_idx, seq_length_per_sample, denoisers, seed, remain_tokens)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save sample-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # shuffle-idx.
            start_time = time.time()

            if separate_last_epoch:
                # num_samples_ = num_samples_from_epochs_minus_one
                pass
            else:
                num_samples_ = sample_idx.shape[0] - 1

            shuffle_idx = _build_shuffle_idx(num_samples_,
                                             sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save shuffle-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
        torch.distributed.get_world_size() //
        torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()))

    # Load mappings.
    start_time = time.time()
    print_rank_0(' > loading doc-idx mapping from {}'.format(
        doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading sample-idx mapping from {}'.format(
        sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading shuffle-idx mapping from {}'.format(
        shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(
        sample_idx.shape[0] - 1))
    print_rank_0('    total number of epochs: {}'.format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx, index_prefix, denoisers, denoisers_info, seq_length_per_sample


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length_per_sample):
    """Based on number of samples and sequence length, calculate how many
    epochs will be needed."""
    return math.ceil((sum(seq_length_per_sample)) / tokens_per_epoch)


def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch):
    """Build an array with length = number-of-epochs * number-of-documents.
    Each index is mapped to a corresponding document."""

    doc_idx = np.tile(documents, num_epochs)
    if num_epochs > 1 and separate_last_epoch:
        np_rng.shuffle(doc_idx[:-len(documents)])
        np_rng.shuffle(doc_idx[-len(documents):])
    else:
        np_rng.shuffle(doc_idx)

    return doc_idx.astype(np.int32)


def _build_sample_idx(sizes, doc_idx, seq_length_per_sample,
                      denoisers, np_rng, remain_tokens):
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 1] is the
    starting offset in that document."""

    assert len(denoisers) == len(seq_length_per_sample)

    denoisers_info = [1860, 1973, 1535, 1819, 2037, 2015, 2048]

    # Correct num_samples by continuously sampling the denoiser.
    # while remain_tokens > 0:
    #     denoiser = np_rng.choice(range(7))
    #     seq_length = denoisers_info[denoiser]
    #     denoisers.append(denoiser)
    #     seq_length_per_sample.append(seq_length)
    #     remain_tokens -= seq_length

    num_samples = len(denoisers)
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Begining offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # -1 because sample_index starts from 1.
        remaining_seq_length = seq_length_per_sample[sample_index - 1]
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
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
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx, denoisers, seq_length_per_sample


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


denoiser_config = {
    0: (3, 0.15, "[R]"),  # R
    1: (8, 0.15, "[R]"),  # R
    2: (3, 0.5, "[X]"),  # X
    3: (8, 0.5, "[X]"),  # X
    4: (64, 0.15, "[X]"),  # X
    5: (64, 0.5, "[X]"),  # X
    6: (0.25, 0.25, "[S]"),  # S
}


def _get_all_denoisers_info(seq_length):
    """ Obtain the necessary information of 7 denoisers, namely the number of tokens, spans and pads. """
    info = dict()
    for k, v in denoiser_config.items():
        if k < 6:
            info[k] = _get_num_tokens_spans_pads_per_denoiser(seq_length, v[0], v[1])
    info[6] = (seq_length - 1, 0, 0)  # S-Denoiser has no span, and naturally no pad
    return info


def _get_num_tokens_spans_pads_per_denoiser(seq_length, mean_span_length, corrupt_rate):
    """ Calculate the number of tokens, spans and pads required for a specific denoiser. """
    assert isinstance(mean_span_length, int), "mean_span_length must be an integer"
    num_tokens = int((seq_length - 1) / (2 * corrupt_rate / mean_span_length + 1))
    num_spans = round(num_tokens * corrupt_rate / mean_span_length)
    num_pads = seq_length - 1 - 2 * num_spans - num_tokens
    if num_pads < 0:
        num_tokens += num_pads
        num_pads = 0
    assert num_tokens + 2 * num_spans + num_pads + 2 == seq_length + 1
    return num_tokens, num_spans, num_pads


def _assign_denoiser(np_rng, num_samples, denoiser_ratios=None):
    """ Assign denoiser to each sample according to the specified ratio. """
    denoiser_ids = list(denoiser_config.keys())
    num_denoisers = len(denoiser_config)

    # Uniform distribution by default when denoiser_ratios is not specified.
    if denoiser_ratios is None:
        denoiser_ratios = [1 / num_denoisers] * num_denoisers

    assert len(denoiser_ratios) == num_denoisers, \
        "the length of denoiser_ratios must be the same as the length of denoiser_config."
    assert abs(sum(denoiser_ratios) - 1) < 1e-6, "the ratios in denoiser_ratios must sum up to 1."

    denoiser_counts = np.round(np.array(denoiser_ratios) * num_samples).astype(int)

    while denoiser_counts.sum() > num_samples:
        decrease_indices = np_rng.choice(num_denoisers, size=denoiser_counts.sum() - num_samples, p=denoiser_ratios)
        denoiser_counts[decrease_indices] -= 1
        denoiser_counts = np.maximum(denoiser_counts, 0)

    while denoiser_counts.sum() < num_samples:
        increase_indices = np_rng.choice(num_denoisers, size=num_samples - denoiser_counts.sum(), p=denoiser_ratios)
        denoiser_counts[increase_indices] += 1

    res = np.repeat(denoiser_ids, denoiser_counts)
    np_rng.shuffle(res)
    return list(res)
