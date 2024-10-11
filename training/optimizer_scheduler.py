import torch
import torch.optim.lr_scheduler as lr_scheduler
from dataclasses import dataclass
import util

@dataclass
class OptimizerScheduler:  
    optimizer: any
    scheduler: any
    
    @classmethod
    def load(clc, args, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=util.LR_START)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=util.SCHEDULER_STEP_SIZE, 
                                        gamma=util.SCHEDULER_GAMMA)
        
        return clc(optimizer=optimizer, scheduler=scheduler)
