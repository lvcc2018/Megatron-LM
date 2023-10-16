
import glob
import json
import os
import time

from torch.utils.data import Dataset

from megatron import print_rank_0, print_rank_last
from tasks.data_utils import build_sample
from tasks.data_utils import build_tokens_types_paddings_from_ids
from megatron.tokenizer.tokenizer import _SentencePieceTokenizer, _MixedTokenizer
from megatron.core import mpu
import torch
import h5py

BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN_ID = 0
NEW_LINE_ID = 13

prefix_format_dict = {
    'en':
        {
            "input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            'noinput': "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
        },
    'zh':
        {
            "input": "下面的指令描述了一个需要完成的任务，并提供了一个输入来作为上下文，请编写一个回复来合理地完成请求。\n\n### 指令：\n{instruction}\n\n### 输入：\n{input}\n\n### 回复：\n",
            "noinput": "下面的指令描述了一个需要完成的任务，请编写一个回复来合理地完成请求。\n\n### 指令：\n{instruction}\n\n### 回复：\n",
        },
    'mix':
        {
            "input": "### 用户(User):\n{instruction}\n{input}\n",
            "noinput": "### 用户(User):\n{instruction}\n",
            "before_user": "### 用户(User):\n",
            "before_assistant": "### 助手(Assistant):\n"
        }
}


class InstructionDataset(Dataset):

    def __init__(self, dataset_name, weight, datapaths, tokenizer, max_seq_length, use_mix_format=False, use_cache=True, rank=0):

        self.dataset_name = dataset_name
        self.weight = weight
        self.use_mix_format = use_mix_format
        print_rank_0(' > building Instruction dataset for {}:'.format(
            self.dataset_name))
        string = '  > paths:'
        for path in datapaths:
            string += ' ' + path
        print_rank_0(string)
        use_hdf5 = False
        file_extension = datapaths[0].split('.')[-1]
        if file_extension == 'hdf5':
            use_hdf5 = True
        self.use_hdf5 = use_hdf5
        if not use_hdf5:
            # load as normal
            self.samples = []
            for datapath in datapaths:
                self.samples.extend(process_single_datapath(datapath, tokenizer,
                                    max_seq_length, use_mix_format, use_cache, rank))

            print_rank_0('  >> total number of samples: {}'.format(
                len(self.samples)))
            self.samples_number = len(self.samples)
        else:
            # load hdf5 data
            assert len(datapaths) == 1, "only support one hdf5 file."
            print_rank_0('   > loading from {}'.format(datapaths[0]))
            self.samples = h5py.File(datapaths[0], 'r')
            self.samples_number = self.samples['samples_number'][()]
            print_rank_0('  >> total number of samples: {}'.format(self.samples_number))

    def __len__(self):
        return self.samples_number

    def __getitem__(self, idx):
        if not self.use_hdf5:
            return self.samples[idx]
        else:
            input_ids = self.samples['input_ids'][idx].tolist()
            labels = self.samples['labels'][idx].tolist()
            loss_mask = self.samples['loss_mask'][idx].tolist()
            segment_ids = self.samples['segment_ids'][idx].tolist()
            sample = {
                'input_ids': input_ids,
                'labels': labels,
                'loss_mask': loss_mask,
                'segment_ids': segment_ids,
                'idx': idx,
            }
            return sample
    
    @classmethod
    def collate_fn(cls, max_seq_length, pad_max_length, user_loss_mask, batch):
        new_batch = {
            'input_ids': [],
            'labels': [],
            'attention_mask': [],
            'loss_mask': [],
        }
        max_len = 0
        for b in batch:
            new_batch['input_ids'].append(b['input_ids'])
            new_batch['labels'].append(b['labels'])
            new_batch['loss_mask'].append(b['loss_mask'])
            if len(b['input_ids']) > max_len:
                max_len = len(b['input_ids'])
        if pad_max_length:
            max_len = max_seq_length
        for i in range(len(batch)):
            current_len = len(new_batch['input_ids'][i])
            new_batch['input_ids'][i] = new_batch['input_ids'][i] + [PAD_TOKEN_ID] * (max_len - current_len)
            new_batch['attention_mask'].append([1] * current_len + [0] * (max_len - current_len))
            new_batch['labels'][i] = new_batch['labels'][i] + [PAD_TOKEN_ID] * (max_len - current_len)
            new_batch['loss_mask'][i] = new_batch['loss_mask'][i] + [0] * (max_len - current_len)
        # print_rank_0(new_batch.keys())
        for key in new_batch:
            new_batch[key] = torch.tensor(new_batch[key], dtype=torch.int64)
        new_batch['loss_mask'] = new_batch['loss_mask'].float()
        new_batch['assistant_loss_mask'] = new_batch['loss_mask'].masked_fill(new_batch['loss_mask'] == 2.0, 0)
        new_batch['loss_mask'] = new_batch['loss_mask'].masked_fill(new_batch['loss_mask'] == 2.0, user_loss_mask)
        if 'segment_ids' not in batch[0]:
            new_batch['attention_mask'] = new_batch['attention_mask'][:, None, :] * new_batch['attention_mask'][:, :, None]
            new_batch['attention_mask'] = torch.tril(torch.ones(new_batch['attention_mask'].shape[1:]), diagonal=0)[None, :, :] * new_batch['attention_mask']
            new_batch['attention_mask'] = (1 - new_batch['attention_mask']).bool()
            new_batch['position_ids'] = torch.arange(new_batch['attention_mask'].shape[1]).long().unsqueeze(0).repeat(new_batch['attention_mask'].shape[0], 1)
        else:
            segment_ids = [b['segment_ids'] for b in batch]
            new_batch['attention_mask'] = torch.ones(len(batch), max_len, max_len, dtype=torch.bool)
            new_batch['position_ids'] = torch.zeros(len(batch), max_len, dtype=torch.int64)
            for i in range(len(batch)):
                c_segment_ids = segment_ids[i]
                next_segment_id = 1
                start_index = 0
                while start_index < max_len:
                    try:
                        end_index = c_segment_ids.index(next_segment_id)
                    except ValueError:
                        end_index = max_len
                    new_batch['position_ids'][i, start_index:end_index] = torch.arange(0, end_index - start_index, dtype=torch.int64)
                    new_batch['attention_mask'][i, start_index:end_index, start_index:end_index] = torch.triu(torch.ones((end_index - start_index, end_index - start_index)), diagonal=1).to(torch.bool)
                    next_segment_id += 1
                    start_index = end_index
        # new_batch['idx'] = []
        # for b in batch:
        #     new_batch['idx'].append(b['idx'])
        return new_batch


def process_single_datapath(datapath, tokenizer, max_seq_length, use_mix_format=False, use_cache=True, rank=0):
    """Read in Instruction files, combine, clean-up, tokenize, and convert to
    samples."""
    if mpu.is_unitialized():
        print_function = print
    else:
        print_function = print_rank_0

    print_function('   > working on {}'.format(datapath))
    start_time = time.time()
    cache_dir = os.path.join(os.path.dirname(datapath), 'cache')
    if rank == 0:
        os.makedirs(cache_dir, exist_ok=True)
    cache_filename = '.'.join(os.path.basename(datapath).split('.')[:-1])
    cache_filename += ".{}sl".format(max_seq_length)
    if use_mix_format:
        cache_filename += '.mix'
    cache_filename += '.cache'
    cache_filepath = os.path.join(cache_dir, cache_filename)
    if os.path.isfile(cache_filepath) and use_cache:
        print_function(f'   > load cache from {cache_filepath}')
        data = torch.load(cache_filepath)
        samples = data['samples']
        exceed_number = data['exceed_number']
        abondan_long_number = data['abondan_long_number']
        abondan_empty_number = data['abondan_empty_number']
        mean_output_number = data['mean_output_number']
    else:
        if not os.path.isfile(cache_filepath) and use_cache:
            print_function(f'   > no cache file of {cache_filepath}, preprocessing data')
        if datapath.endswith('.json'):
            with open(datapath) as f:
                datas = json.load(f)
        elif datapath.endswith('.jsonl'):
            datas = []
            with open(datapath) as f:
                for linenum, line in enumerate(f):
                    line = line.strip()
                    if line == '':
                        continue
                    datas.append(json.loads(line))
        else:
            raise ValueError("the type of {datapath} is not supported.")
        samples = []
        exceed_number = 0
        abondan_long_number = 0
        abondan_empty_number = 0
        output_number = []
        if use_mix_format:
            print_function('   > use mix format')
            current_format = prefix_format_dict['mix']
        elif 'zh' in datapath:
            print_function('   > use chinese format')
            current_format = prefix_format_dict['zh']
        elif 'en' in datapath:
            # use english format
            print_function('   > use english format')
            current_format = prefix_format_dict['en']
        else:
            print_function('   > no `zh` or `en` in filename, use mix format')
            current_format = prefix_format_dict['mix']
        prefix_format_input = current_format['input']
        prefix_format_noinput = current_format['noinput']
        before_user = current_format.get('before_user', '')
        before_assistant = current_format.get('before_assistant', '')
        for d_index, d in enumerate(datas):
            sample = {}
            sample['index'] = d_index
            if 'input' not in d or d['input'] == '':
                prefix = prefix_format_noinput.format(**d)
            else:
                prefix = prefix_format_input.format(**d)
            sample['input'] = prefix
            target = d['output']
            if not isinstance(target, list):
                target = target.split('</s>')
            target = [t.strip() for t in target if t.strip() != '']
            if len(target) == 0:
                # remove empty output
                abondan_empty_number += 1
                continue
            sample['output'] = target
            # ["This is a response.", "This is a response"]
            # input: User: This is a response.</s>Assistant: This is a response.
            # labels: User: This is a response.</s>Assistant: This is a response.</s>
            # full_text = ""
            # full_text += prefix
            # for o_index, o in enumerate(target):
            #     if o_index % 2 == 0:
            #         o = before_assistant + o
            #     else:
            #         o = before_user + o
            #     full_text += o
            if isinstance(tokenizer, _MixedTokenizer):
                input_ids = []
                labels = []
                loss_mask = []
                tokenized_prefix = tokenizer.tokenize(prefix, bos=True, eos=False)
                prefix_len = len(tokenized_prefix)
                input_ids.extend(tokenized_prefix)
                labels.extend(tokenized_prefix[1:])
                loss_mask.extend([0] * len(labels))
                for o_index, o in enumerate(target):
                    before_o = ""
                    if o_index % 2 == 0:
                        before_o += before_assistant
                    else:
                        before_o += before_user
                    if before_o != "":
                        tokenized_before = tokenizer.tokenize(before_o, bos=False, eos=False)
                        # input_ids.extend(tokenized_before)
                        # labels.extend(tokenized_before)
                        loss_mask.extend([0] * len(tokenized_before))
                    tokenized_o = tokenizer.tokenize(before_o + o, bos=False, eos=True)
                    input_ids.extend(tokenized_o[:-1] + [NEW_LINE_ID]) # no eos in input ids
                    labels.extend(tokenized_o)
                    if o_index % 2 == 0:
                        # assistant
                        loss_mask.extend([1] * (len(tokenized_o) - len(tokenized_before)))
                    else:
                        # user
                        loss_mask.extend([2] * (len(tokenized_o) - len(tokenized_before)))
                input_ids = input_ids[:-1]
            elif isinstance(tokenizer, _SentencePieceTokenizer):
                input_ids = []
                labels = []
                loss_mask = []
                tokenized_prefix = tokenizer.tokenize(BOS_TOKEN + prefix)
                prefix_len = len(tokenized_prefix)
                input_ids.extend(tokenized_prefix)
                labels.extend(tokenized_prefix[1:])
                loss_mask.extend([0] * len(labels))
                for o_index, o in enumerate(target):
                    before_o = ""
                    if o_index % 2 == 0:
                        before_o += before_assistant
                    else:
                        before_o += before_user
                    if before_o != "":
                        tokenized_before = tokenizer.tokenize(before_o)
                        # input_ids.extend(tokenized_before)
                        # labels.extend(tokenized_before)
                        loss_mask.extend([0] * len(tokenized_before))
                    tokenized_o = tokenizer.tokenize(before_o + o + EOS_TOKEN)
                    input_ids.extend(tokenized_o[:-1] + [NEW_LINE_ID]) # no eos in input ids
                    labels.extend(tokenized_o)
                    if o_index % 2 == 0:
                        # assistant
                        loss_mask.extend([1] * (len(tokenized_o) - len(tokenized_before)))
                    else:
                        # user
                        loss_mask.extend([2] * (len(tokenized_o) - len(tokenized_before)))
                input_ids = input_ids[:-1]
            
            assert len(input_ids) == len(labels), f"{d_index} in {datapath} have different length of input ids and labels."
            if len(input_ids) > max_seq_length:
                exceed_number += 1
            sample['input_ids'] = input_ids[:max_seq_length]
            sample['labels'] = labels[:max_seq_length]
            sample['loss_mask'] = loss_mask[:max_seq_length]
            if sum(sample['loss_mask']) == 0:
                abondan_long_number += 1
                continue
            sample['prefix_len'] = prefix_len - 1
            samples.append(sample)
            output_number.append(len(target))
            # print_rank_last(f"Index {len(samples)}:")
            # print_rank_last(sample['input'])
            # for t_index, t in enumerate(target):
            #     print_rank_last(f"target {t_index}: {t}")
            # # for input_id, label in zip(sample['input_ids'], sample['labels']):
            # #     print_rank_last(f"{input_id} {label}")
            # print_rank_last("Detokenize input_ids:")
            # print_rank_last(sample['input_ids'])
            # print_rank_last(tokenizer.detokenize(sample['input_ids'], skip_special_tokens=False))
            # chunk_labels = []
            # new_chunk = True
            # for index, mask in enumerate(sample['loss_mask']):
            #     if mask == 0:
            #         new_chunk = True
            #         continue
            #     if new_chunk:
            #         chunk_labels.append([])
            #         new_chunk = False
            #     chunk_labels[-1].append(sample['labels'][index])
            # for c_index, c in enumerate(chunk_labels):
            #     if len(c) > 0:
            #         print_rank_last(f"label {c_index}:")
            #         print_rank_last(c)
            #         print_rank_last(f"{tokenizer.detokenize(c, skip_special_tokens=False)}")
            # if len(samples) == 3:
            #     break
        # quit()
        mean_output_number = sum(output_number) / len(output_number)
        print_function("   > save cache file to {}".format(cache_filepath))
        if rank == 0:
            torch.save({
                'samples': samples,
                'exceed_number': exceed_number,
                'abondan_long_number': abondan_long_number,
                'abondan_empty_number': abondan_empty_number,
                'mean_output_number': mean_output_number,
            }, cache_filepath)
        
    elapsed_time = time.time() - start_time
    print_function('    > processed {} samples in {:.2f} seconds, {} samples exceed max sequence length.'.format(len(samples), elapsed_time, exceed_number))
    print_function('    > {} samples abondan by empty output. {} samples abondan by long prefix. mean multi turn number is {:.2f}.'.format(abondan_empty_number, abondan_long_number, mean_output_number))
    # quit()

    return samples
    
    
    