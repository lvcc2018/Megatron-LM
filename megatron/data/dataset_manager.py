import torch
import numpy as np
from collections import Counter
from megatron import print_rank_0

class WeightScheduler():
    def __init__(self, weights, names, global_batch_size, train_iters, split_name):
        print_rank_0(f"building weight scheduler for dataset manager for {split_name}")
        self.names = names
        normalized_weights = []
        for w in weights:
            if '-' not in w:
                # constant
                c_weights = [float(w)] * train_iters
            else:
                # linear
                linear_args = w.split('-')
                max_weights = float(linear_args[1])
                min_weights = float(linear_args[0])
                if len(linear_args) >= 3:
                    linear_iters = int(float(linear_args[2]) * train_iters)
                else:
                    linear_iters = train_iters
                print_rank_0(f"linear {linear_iters} train {train_iters}")
                assert linear_iters <= train_iters, f"linear {linear_iters} train {train_iters}"
                interval = (max_weights - min_weights) / (linear_iters - 1)
                c_weights = list(range(0, linear_iters))
                c_weights = [w * interval + min_weights for w in c_weights]
                if train_iters > linear_iters:
                    c_weights.extend([c_weights[-1]] * (train_iters - linear_iters))
            assert len(c_weights) == train_iters
            normalized_weights.append(c_weights)
        self.weights = []
        for i in range(train_iters):
            w = [nw[i] for nw in normalized_weights]
            self.weights.append([nw / sum(w) for nw in w])
        self.global_batch_size = global_batch_size
        self.train_iters = train_iters
        self.batches = self.generate_each_batch()
        self.split_name = split_name
    
    def generate_each_batch(self):
        batches = []
        start_indexes = [0] * len(self.names)
        for i in range(self.train_iters):
            weights = self.weights[i]
            numbers = [round(self.global_batch_size * w) for w in weights]
            current_batchs = []
            for index, num in enumerate(numbers):
                if len(current_batchs) + num <= self.global_batch_size:
                    current_batchs.extend(zip([index] * num, range(start_indexes[index], start_indexes[index] + num)))
                else:
                    num = self.global_batch_size - len(current_batchs)
                    current_batchs.extend(zip([index] * num, range(start_indexes[index], start_indexes[index] + num)))
                start_indexes[index] += num
            if len(current_batchs) < self.global_batch_size:
                num = self.global_batch_size - len(current_batchs)
                index = current_batchs[-1][0]
                current_batchs.extend(zip([index] * num, range(start_indexes[index], start_indexes[index] + num)))
                start_indexes[index] += num
            batches.append(current_batchs)
        return batches
            
    def get_each_sample_number(self):
        numbers = [0] * len(self.names)
        for dataset_index, in_data_index in self.batches[-1]:
            numbers[dataset_index] = in_data_index + 1
        # for b in self.batches:
        #     each_number = Counter(b)
        #     for key, value in each_number.items():
        #         numbers[key] += value
        samples_weights = [n / sum(numbers) for n in numbers]
        log_str = f"actual samples weight for each dataset in {self.split_name}: "
        for name, weight in zip(self.names, samples_weights):
            log_str += f"{name}: {weight:.4f}, "
        print_rank_0(log_str)
        return numbers
    
    def get_batch(self, batch_index):
        return self.batches[batch_index]

class DatasetManager(torch.utils.data.Dataset):
    def __init__(self, names, datasets, weight_scheduler, global_batch_size, train_iters):
        self.names = names
        self.datasets = datasets
        self.weight_scheduler = weight_scheduler
        self.global_batch_size = global_batch_size
        self.train_iters = train_iters
    
    def __len__(self):
        return sum([len(d) for d in self.datasets])
    
    def __getitem__(self, idx):
        batch_index = idx // self.global_batch_size
        in_batch_index = idx % self.global_batch_size
        dataset_index, in_data_index = self.weight_scheduler.get_batch(batch_index)[in_batch_index]
        data = self.datasets[dataset_index][in_data_index]
        # source = self.names[dataset_index]
        data['source'] = np.array([dataset_index], dtype=np.int64)
        # print(f"Idx: {idx}, Batch: {batch_index}, In batch: {in_batch_index}, Source: {source}, In data index: {in_data_index}")
        return data
    