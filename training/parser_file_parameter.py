from dataclasses import dataclass
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import util

@dataclass
class Parser:
    args: argparse
    
    @classmethod
    def init(clc):
        if util.TRAIN_MODEL:
            description = util.DESCR_TRAINING
        else:
            description = util.DESCR_TRAINED_MODEL
            
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('--epoch_max' , '-E', type=int, default=util.EPOCH_MAX)
        parser.add_argument('--epoch_from', '-EF', type=int, default=util.EPOCH_FROM)
        parser.add_argument('--gpu', '-G', type=int, default=util.GPU)
        args = parser.parse_args()
        return clc(args=args)
        
@dataclass            
class File:
    save_name: str
    checkpoint_model_file: str
    checkpoint_optim_file: str
    best_model_file: str
    final_model_file: str
    log_file: str
    save_pred_mask_dir_train: str
    save_pred_mask_dir_test: str
    
    @classmethod
    def build(clc, args, model_name, dataset_name, spectrum_name):
        save_name = Path(util.SAVE.PRED_DIR) / Path(f'{model_name}_{dataset_name}_{spectrum_name}')
        os.makedirs(save_name, exist_ok=True)
        
        checkpoint_model_file = save_name / Path(util.FILE.CHECKPOINT_MODEL_FILE)
        checkpoint_optim_file = save_name / Path(util.FILE.CHECKPOINT_OPTIM_FILE)
        best_model_file       = save_name / Path(util.FILE.BEST_MODEL_FILE)  
        final_model_file      = save_name / Path(util.FILE.FINAL_MODEL_FILE)
        log_file              = save_name / Path(util.FILE.LOG_FILE)
        save_pred_mask_dir_train    = save_name / Path(util.FILE.SAVE_PRED_MASK_DIR_TRAIN)
        save_pred_mask_dir_test    = save_name / Path(util.FILE.SAVE_PRED_MASK_DIR_TEST)

        if not os.path.isdir(save_pred_mask_dir_train):
            os.makedirs(save_pred_mask_dir_train)
            
        if not os.path.isdir(save_pred_mask_dir_test):
            os.makedirs(save_pred_mask_dir_test)

        print('| training %s on GPU #%d with pytorch' % (save_name, args.gpu))
        print('| from epoch %d / %s' % (args.epoch_from, args.epoch_max))
        print('| model will be saved in: %s' % save_name)
        
        return clc(save_name=save_name, checkpoint_model_file=checkpoint_model_file, 
                   checkpoint_optim_file=checkpoint_optim_file,
                   best_model_file=best_model_file, final_model_file=final_model_file, 
                   log_file=log_file, save_pred_mask_dir_train=save_pred_mask_dir_train,
                   save_pred_mask_dir_test=save_pred_mask_dir_test)
             
@dataclass       
class Parameter:
    model_name: str
    dataset_name: str
    spectrum_name: str
    save_name: str
    