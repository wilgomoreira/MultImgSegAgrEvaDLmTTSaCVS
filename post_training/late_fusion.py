from dataclasses import dataclass
from metrics import Metrics
import numpy as np
import util

@dataclass
class LateFusion:
    name: str
    model: str
    database: str
    spectrum: str
    value: np
             
    @classmethod
    def simple_mean(clc, preds): 
        name = util.L_FUSION_SIMPLE_MEAN
        spectrum = name
        
        spec_preds = []
        for pred in preds:
            if pred.spectrum != util.CHOSEN_FUSION.lower():
              spec_preds.append(pred)  
        
        model = spec_preds[0].model
        database = spec_preds[0].database
    
        sum = 0
        for spec_pred in spec_preds:
            sum += spec_pred.value
            
        mean = sum / len(spec_preds)
        fusion = mean
        
        return clc(name=name, model=model, database=database, spectrum=spectrum, value=fusion)
    
    @classmethod
    def weighted_mean(clc, preds, mask):
        name = util.L_FUSION_WEIGHTED_MEAN
        spectrum = name
        
        spec_preds = []
        for pred in preds:
            if pred.spectrum != util.CHOSEN_FUSION.lower():
              spec_preds.append(pred)  
        
        model = spec_preds[0].model
        database = spec_preds[0].database
        
        metrs = []
        
        for spec_pred in spec_preds:
            metrs.append(Metrics.evaluate(spec_pred, mask))
        
        weighted_metr = util.MET_LATE_FUSION.WEIGHT_F1S
        
        weights = []
        for metr in metrs:
           weights.append(getattr(metr, weighted_metr))
        
        sum_prods = 0
        num_samples = len(spec_preds[0].value)
        
        for weight, spec_pred in zip(weights, spec_preds):
            weight = weight.reshape(num_samples, 1, 1, 1)
            sum_prods += weight * spec_pred.value
        
        sum_weights = 0
        for weight in weights:
            sum_weights += weight
            
        sum_weights = sum_weights.reshape(num_samples, 1, 1, 1)
        weighted_mean = sum_prods / sum_weights
        fusion = weighted_mean
        
        return clc(name=name, model=model, database=database, spectrum=spectrum, value=fusion)
    
    