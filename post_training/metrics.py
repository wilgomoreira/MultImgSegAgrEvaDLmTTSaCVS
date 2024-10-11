from dataclasses import dataclass
import torch
import numpy as np
from image import Image
import util

@dataclass
class Metrics:
    name: str
    model: str
    database: str
    spectrum: str
    
    f1s: np
    ious: np
    mean_f1s: np
    mean_ious: np
   
    preds: Image
    masks: Image
    inputs: Image
    
    @staticmethod
    def for_all_images(all_images):
        inputs, masks, preds = all_images
        all_metrs = []
        
        for input, mask, pred in zip(inputs, masks, preds):
            all_metrs.append(Metrics.evaluate(preds=pred, masks=mask, inputs=input))
         
        return all_metrs
    
    @classmethod   
    def evaluate(clc, preds, masks, inputs):
        f1s, ious = _calculate_metrics(preds=preds, masks=masks)
        mean_f1s, mean_ious = _mean_metrics(f1s=f1s, ious=ious)
        
        return clc(name=preds.name, model=preds.model, database=preds.database, 
                   spectrum=preds.spectrum, f1s=f1s, ious=ious, mean_f1s=mean_f1s, 
                   mean_ious=mean_ious, inputs=inputs, masks=masks, preds=preds)
           

def _calculate_metrics(preds, masks):
    probs = preds.value
    masks = masks.value
    
    #eces = _ece_score_metric(pred=probs, mask=masks)
    
    if (util.CALIBRATION == False) and (util.CONDICIONAL_RANDOM_FIELD == False) and (util.BAYES == False):
        probs = util.sigmoid(probs)
    
    probs = util.thresholding(probs=probs)
    f1s, ious = _perfomance_metrics(bin_preds=probs, masks=masks)
        
    return f1s, ious
        
def _perfomance_metrics(bin_preds, masks):
    NOT_ZERO = 1e-7
    
    if isinstance(bin_preds, np.ndarray):
        bin_preds = torch.from_numpy(bin_preds)

    f1s, ious = [], []

    for bin_pred, mask in zip(bin_preds, masks):
        tp = torch.sum((bin_pred == 1) & (mask == 1)).item()
        fp = torch.sum((bin_pred == 1) & (mask == 0)).item()
        tn = torch.sum((bin_pred == 0) & (mask == 0)).item()
        fn = torch.sum((bin_pred == 0) & (mask == 1)).item()
        
        acc = (tp + tn) / (tp + fp + tn + fn + NOT_ZERO)
        precision = tp / (tp + fp + NOT_ZERO)
        recall = tp / (tp + fn + NOT_ZERO)
        f1_score = (2 * precision * recall) / (precision + recall + NOT_ZERO)

        intersection = torch.sum((bin_pred == 1) & (mask == 1)).item()
        union = torch.sum((bin_pred == 1) | (mask == 1)).item()
        iou = intersection / (union + NOT_ZERO)
        
        if acc == 0:
            acc = NOT_ZERO
        if f1_score == 0:
            f1_score = NOT_ZERO
        if iou == 0:
            iou = NOT_ZERO
        
        f1s.append(f1_score)
        ious.append(iou)
    
    f1s = np.array(f1s)
    ious = np.array(ious)   
    
    return f1s, ious

def _mean_metrics(f1s, ious):
    mean_f1s = np.mean(f1s)
    mean_ious = np.mean(ious) 
    #mean_eces = np.mean(eces) 
    
    return mean_f1s, mean_ious       

def _ece_score_metric(pred, mask):
    
    pred = np.array(pred)

    pred_ece = []
    for pred_sample, mask_sample in zip(pred, mask):
        #transform to vector
        pred_sample_flat = pred_sample.flatten()
        mask_sample_flat = mask_sample.flatten()
        
        value_ece = ece_evaluate(pred=pred_sample_flat, mask=mask_sample_flat)
        value_ece = np.array(value_ece)
        pred_ece.append(value_ece)
        
    return pred_ece
     
def ece_evaluate(pred, mask):
    
    n_bins = util.NUM_BINS
    
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0  # Initialize ECE
    
    # Calculate ECE using bins
    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        # Find the indices of probabilities that fall into the current bin
        bin_indices = np.where((pred > bin_lower) & (pred <= bin_upper))[0]
        
        # Continue to the next bin if no samples found for the current bin
        if len(bin_indices) == 0:
            continue
        
        # Calculate the accuracy and the confidence for the current bin
        bin_acc = np.mean(mask[bin_indices] == (mask[bin_indices] > 0.5))
        bin_conf = np.mean(pred[bin_indices])
        
        # Calculate the absolute difference between accuracy and confidence
        bin_error = np.abs(bin_acc - bin_conf)
        
        # Weight the bin error by the proportion of samples in the bin
        bin_weight = len(bin_indices) / len(pred)
        
        # Add the weighted error to the total ECE
        ece += bin_error * bin_weight
    
    return ece


    
    
        
        
            
        
        

        
 