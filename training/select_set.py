import torch
import util
from dataclasses import dataclass

@dataclass
class SelectSet:
    train_dataloader: torch
    test_dataloader: torch
    
    @classmethod
    def separate_them(clc, dataset_name, load_dat):
        batch_train = util.BATCH_SIZE_TRAIN
        batch_test = util.BATCH_SIZE_TEST
        shuffle_train = util.SUFFLE_TRAIN
        shuffle_test = util.SUFFLE_TEST
        
        match dataset_name.lower():
            case util.T1:
                train_dataset = torch.utils.data.ConcatDataset([load_dat.loader_val, 
                                                                load_dat.loader_esac]) 
                train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                               batch_size=batch_train, 
                                                               shuffle=shuffle_train)
                test_dataloader = torch.utils.data.DataLoader(load_dat.loader_qbaixo, 
                                                              batch_size=batch_test, 
                                                              shuffle=shuffle_test)
            case util.T2:
                train_dataset = torch.utils.data.ConcatDataset([load_dat.loader_val, 
                                                                load_dat.loader_qbaixo])
                train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                               batch_size=batch_train, 
                                                               shuffle=shuffle_train)
                test_dataloader = torch.utils.data.DataLoader(load_dat.loader_esac, 
                                                              batch_size=batch_test, 
                                                              shuffle=shuffle_test)              
            case util.T3:
                train_dataset = torch.utils.data.ConcatDataset([load_dat.loader_esac, 
                                                                load_dat.loader_qbaixo])
                train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                               batch_size=batch_train, 
                                                               shuffle=shuffle_train)
                test_dataloader = torch.utils.data.DataLoader(load_dat.loader_val, 
                                                              batch_size=batch_test, 
                                                              shuffle=shuffle_test)
        
        return clc(train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    
        
        
        
        
        