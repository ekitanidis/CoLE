import math
import numpy as np


class CosineScheduler(object):
    
    def __init__(self, optimizer, warmup_iters, warmup_lr, total_iters, base_lr, final_lr, constant_predictor_lr=True, **args):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        decay_iters = total_iters - warmup_iters
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iters)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(np.pi * np.arange(decay_iters) / decay_iters))
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0 
        
    def step(self):
        for param_group in self.optimizer.param_groups:
            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            elif self.constant_predictor_lr and param_group['name'] == 'encoder':
                lr = param_group['lr'] = self.lr_schedule[self.iter]            
        self.iter += 1
        return lr
    
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        

        
class LinearScheduler(object):
    
    def __init__(self, optimizer, warmup_iters, total_iters, base_lr, **args):
        self.base_lr = base_lr
        decay_iters = total_iters - warmup_iters
        warmup_lr_schedule = np.linspace(0, base_lr, warmup_iters)
        decay_lr_schedule = np.linspace(base_lr, 0, decay_iters)
        self.lr_schedule = np.concatenate((warmup_lr_schedule, decay_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        
    def step(self):
        lr = self.lr_schedule[self.iter]
        self.iter += 1
        return lr
    
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
