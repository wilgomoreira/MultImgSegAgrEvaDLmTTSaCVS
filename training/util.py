import torch
import numpy as np
import time

TRAIN_MODEL = False # If you want to train the models, change for 
TRAIN_DATALOADER = "train_dataloader"
TEST_DATALOADER = "test_dataloader"
DATALOADER_RESULTS = TRAIN_DATALOADER   # what dataset you want the results with train_model = False

USE_CALIBRATION = False
USE_CRF = False
OUT_RESULTS  = "training/calibration_results.txt"
N_BINS = 10

T1, T2, T3 = "t1", "t2", "t3"
SPECTRUMS = ["RGB", "NDVI", "GNDVI", "EARLY_FUSION"]
DATASETS = ['t1','t2','t3']
MODELS = ["segnet", "deeplab"] 
RGB = "RGB"
NDVI = "NDVI"
GNDVI = "GNDVI"
EARLY_FUSION = "EARLY_FUSION"
SEGNET = "segnet"
DEEPLAB = "deeplab"

DESCR_TRAINING = "Training with Pytorch"
DESCR_TRAINED_MODEL = "Using Trained Models"

NUM_CLASSES = 1
NDVI_SHAPE = 3
GNDVI_SHAPE = 3
NDVI_UNSQUEEZE = 1
GNDVI_UNSQUEEZE = 1

EPOCH_MAX = 100
EPOCH_FROM = 0
GPU = 0

LR_START = 0.001
RUNNING_LOSS = 0
MAPA_LOCATION = {'cuda:0':'cuda:1'} 
MAPA_LOC_DEVICE = 'cuda:0'
BATCH_SIZE_TRAIN = 30
BATCH_SIZE_TEST = 3
SUFFLE_TRAIN = True
SUFFLE_TEST = False

THRESHOLD = 0.5
INPUT_RGB_CHAN = 3
INPUT_NDVI_CHAN = 1
INPUT_GNDVI_CHAN = 1
    
DEEPLAB_VERSION = 'deeplabv3_resnet50'
DEEPLAB_OUTPUT_STRIDE = 8
SCHEDULER_STEP_SIZE = 50
SCHEDULER_GAMMA = 0.1
ROUND_PERC = 2
MEAN_DURATION = 3

class FILE:
    CHECKPOINT_MODEL_FILE = 'tmp.pth'
    CHECKPOINT_OPTIM_FILE = 'tmp.optim'
    BEST_MODEL_FILE = 'best.pth'
    FINAL_MODEL_FILE = 'final.pth'
    LOG_FILE = 'log.txt'
    SAVE_PRED_MASK_DIR_TRAIN = 'pred_masks_train'
    SAVE_PRED_MASK_DIR_TEST = 'pred_masks_test'

class DATASET_ROOT:      
    VALDOEIRO = "/mnt/0D6BEAD6291820B7/Wilgo/Datasets/greenaI_split/valdoeiro/"
    ESAC = "/mnt/0D6BEAD6291820B7/Wilgo/Datasets/greenaI_split/esac/"
    QBAIXO = "/mnt/0D6BEAD6291820B7/Wilgo/Datasets/greenaI_split/qbaixo/"
        
class DATASET:
    LABELS = ['Field','Corn']
    DATASET_MODE = "DISK"
    TARGET_SIZE = (240, 240)
    RANDOM_HORIN_FLIP = 0.8
    SET = "altum"
    RGB_DIR = "images"
    MASK_DIR = "masks"
    SENSORS = ['images','masks']    

class SAVE:
    OVERAL_PERFM = "training/global_perfm.txt"
    PRED_DIR = 'training/weights'
    

class FUNC:     
    class SHAPE_GPU:
        def evaluate(args, rgb, ndvi, gndvi, mask):
            if len(ndvi.shape) <= NDVI_SHAPE:
                ndvi = ndvi.unsqueeze(NDVI_UNSQUEEZE)   
            
            if len(gndvi.shape) <= GNDVI_SHAPE:
                gndvi = gndvi.unsqueeze(GNDVI_UNSQUEEZE) 
                       
            if args.gpu >= GPU:
                rgb = rgb.cuda(args.gpu)            
                ndvi = ndvi.float().cuda(args.gpu)  
                gndvi = gndvi.float().cuda(args.gpu)
                mask = mask.cuda(args.gpu)  
            return rgb, ndvi, gndvi, mask
    
    class DECISION: 
        def take_a_one(probs):
            bin_preds = (probs > THRESHOLD).float()
            return bin_preds    
        
        def choose_input_mode(spec_name, rgb, ndvi, gndvi):
            match spec_name:   
                case "RGB":
                    input_mod = rgb  
                    
                case "NDVI":
                    input_mod = ndvi  
                    
                case "GNDVI":
                    input_mod = gndvi
                    
                case "EARLY_FUSION":
                    image_fusion = torch.cat([rgb, ndvi, gndvi], dim=1)
                    input_mod = image_fusion
    
            return input_mod
    
        def time_for_model(model, input_mod):  
            t0 = time.time()
            output, _ = model(input_mod)
            t1 = time.time()
            
            return t0, t1, output
           
    class METRIC:
        def compute_performance_metrics(preds, mask):
            NOT_ZERO = 1e-7

            tp = torch.sum((preds == 1) & (mask == 1)).item()
            fp = torch.sum((preds == 1) & (mask == 0)).item()
            tn = torch.sum((preds == 0) & (mask == 0)).item()
            fn = torch.sum((preds == 0) & (mask == 1)).item()
        
            acc = (tp + tn) / (tp + fp + tn + fn + NOT_ZERO)
            precision = tp / (tp + fp + NOT_ZERO)
            recall = tp / (tp + fn + NOT_ZERO)
            f1_score = (2 * precision * recall) / (precision + recall + NOT_ZERO)

            intersection = torch.sum((preds == 1) & (mask == 1)).item()
            union = torch.sum((preds == 1) | (mask == 1)).item()
            iou = intersection / (union + NOT_ZERO)
            
            metrics = acc, f1_score, iou, precision, recall
            return metrics
        
        def mean__metrics_perc(metrs, time_duration=None):
            acc, f1_score, iou, precision, recall = metrs
        
            if time_duration:
                mean_duration = round(np.mean(time_duration), MEAN_DURATION)
            else:
                mean_duration = None
        
            mean_accuracy = np.mean(acc)
            mean_f1_score = np.mean(f1_score)
            mean_iou = np.mean(iou)
            mean_precision = np.mean(precision)
            mean_recall = np.mean(recall)
           
            mean_acc_perc = round(mean_accuracy * 100, ROUND_PERC)
            mean_f1_scor_perc = round(mean_f1_score * 100, ROUND_PERC)
            mean_iou_perc = round(mean_iou * 100, ROUND_PERC)
            mean_precis_perc = round(mean_precision * 100, ROUND_PERC)
            mean_recall_perc = round(mean_recall * 100, ROUND_PERC)
            
            return (mean_duration, mean_acc_perc, mean_precis_perc, mean_recall_perc, 
                    mean_f1_scor_perc, mean_iou_perc)
            
       
        
        
       
        
        
        
     
         


