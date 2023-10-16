import glob
import json
import os
import time

from torch.utils.data import WeightedRandomSampler
from torch import Generator
import numpy as np
import random
from megatron import print_rank_0, print_rank_last

class InfiniDataloader():
    '''
    Convert a DataLoader to infinite dataloader
    '''
    def __init__(self, dataloader, seed, name):
        self.dataloader = dataloader
        self.iter_dataloader = iter(dataloader)
        self.epoch = 0
        self.cnt = 0
        self.seed = seed
        self.name = name
    
    def __next__(self):
        try:
            data = next(self.iter_dataloader)
            self.cnt += 1
        except StopIteration as e:
            # new epoch
            self.epoch += 1
            self.cnt = 0
            if hasattr(self.dataloader.sampler, 'set_epoch'):
                self.dataloader.sampler.set_epoch(self.seed + self.epoch)
            self.iter_dataloader = iter(self.dataloader)
            data = next(self.iter_dataloader)
            self.cnt += 1
        return data
    
    def get_states(self):
        return {self.name: self.epoch + self.cnt / len(self.dataloader)}

class ConstantRateScheduler:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        
    def step(self):
        pass
    
    def get_rate(self):
        return self.sample_rate
        

class DataLoaderManager():
    '''
    Multiple DataLoaders
    '''
    def __init__(self, dataloaders, rank, world_size, gradient_accumulation_steps, seed, sample=True, consumed_iterations=0):
        '''
        dataloaders: list of (dataloader, seed, name, sample_rate_scheduler)
        sample: whether to sample dataset. Be False on validation
        '''
        self.sample = sample
        self.dataloaders = [InfiniDataloader(*d[0:3]) for d in dataloaders]
        self.sample_rate_schedulers = [d[3] for d in dataloaders]
        self.rank = rank
        self.world_size = world_size
        self.sample_index = 0
        self.cnt = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.np_rng = np.random.RandomState(seed=seed)
        if consumed_iterations > 0:
            # for resuming
            self.np_rng.randint(0, 2**16 - 1, consumed_iterations)
        
    def sample_dataset(self):
        # if self.rank == 0:
        #     sample_rates = [s.get_rate() for s in self.sample_rate_schedulers]
        #     seed = random.randint(0, 2**16 - 1)
        #     generator = Generator()
        #     generator.manual_seed(seed)
        #     self.sample_index = list(WeightedRandomSampler(sample_rates, 1, replacement=True, generator=generator))[0]
        # else:
        #     self.sample_index = None
        # if self.world_size > 1:
        #     if self.rank == 0:
        #         sample_index_tensor = torch.LongTensor([self.sample_index]).cuda()
        #     else:
        #         sample_index_tensor = torch.LongTensor([0]).cuda()
        #     torch.distributed.broadcast(sample_index_tensor, 0)
        #     sample_index = sample_index_tensor.item()
        #     if self.rank == 0:
        #         assert sample_index == self.sample_index
        #     self.sample_index = sample_index
        sample_rates = [s.get_rate() for s in self.sample_rate_schedulers]
        seed = self.np_rng.randint(0, 2**16 - 1)
        generator = Generator()
        generator.manual_seed(seed)
        self.sample_index = list(WeightedRandomSampler(sample_rates, 1, replacement=True, generator=generator))[0]
        # print(f"rank {self.rank}: seed is {seed}, sample index is {self.sample_index}")
    
    def sample_next(self):
        """Used on train

        Returns:
            _type_: _description_
        """
        if self.cnt % self.gradient_accumulation_steps == 0:
            self.sample_dataset()
        self.cnt += 1
        # return self.sample_index, next(self.dataloaders[self.sample_index])
        return self.dataloaders[self.sample_index].name, next(self.dataloaders[self.sample_index])
    
    def sequence_next(self):
        """Used on dev
        """
        name = self.dataloaders[self.sample_index].name
        data = next(self.dataloaders[self.sample_index])
        current_dataloader = self.dataloaders[self.sample_index].dataloader
        passed_cnt = self.dataloaders[self.sample_index].cnt
        if passed_cnt % len(current_dataloader) == 0:
            # after one epoch
            self.sample_index += 1
            self.sample_index = self.sample_index % len(self.dataloaders)
        return name, data
    
    def one_epoch_iters(self):
        iters = 0
        for dataloader in self.dataloaders:
            iters += len(dataloader.dataloader)
        return iters
            
    
    def __next__(self):
        if self.sample:
            return self.sample_next()
        else:
            return self.sequence_next()
            
    
    def get_states(self):
        states = {}
        for d in self.dataloaders:
            states.update(d.get_states())
        return states
    
    def __iter__(self):
        return self
        
        
    
    
    