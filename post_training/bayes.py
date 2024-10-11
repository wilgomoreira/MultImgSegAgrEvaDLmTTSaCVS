import numpy as np
import util
import copy

class Bayes:
    
    @staticmethod
    def combine_likehoods_models(preds):
        
        half = len(preds) // 2
        preds_model1 = preds[:half] 
        preds_model2 = preds[half:] 
        
        preds_bayes = []
        for pred_model1, pred_model2 in zip(preds_model1, preds_model2):
            value_baye = _bayes_samples_weights(pred_model1=pred_model1, pred_model2=pred_model2)
            value_baye = np.array(value_baye)
            pred_bayes = copy.copy(pred_model1)
            pred_bayes.model = util.MODEL_NAME_BAYES
            pred_bayes.value = value_baye
            preds_bayes.append(pred_bayes)

        return preds_bayes
  
    
def _bayes_samples_weights(pred_model1, pred_model2):
    
    pred_model1 = pred_model1.value
    pred_model2 = pred_model2.value
    
    if (util.CALIBRATION == False) and (util.CONDICIONAL_RANDOM_FIELD == False):
        pred_model1 = util.sigmoid(pred_model1)
        pred_model2 = util.sigmoid(pred_model2)
    
    preds_sample = []
    for pred_model1_sample, pred_model2_sample in zip(pred_model1, pred_model2):
        pred_sample = pred_model1_sample * util.BAYES_WEIGHT1 + pred_model2_sample * util.BAYES_WEIGHT2
        preds_sample.append(pred_sample)
    return np.array(preds_sample)
    
      
def _bayes_samples_a_piori(pred_model1, pred_model2):
    
    pred_model1 = pred_model1.value
    pred_model2 = pred_model2.value
    
    if (util.CALIBRATION == False) and (util.CONDICIONAL_RANDOM_FIELD == False):
        pred_model1 = util.sigmoid(pred_model1)
        pred_model2 = util.sigmoid(pred_model2)
    
    preds_sample = []
    for pred_model1_sample, pred_model2_sample in zip(pred_model1, pred_model2):
        MULT_ALL = pred_model1_sample * pred_model2_sample * util.A_PRIOR_PROB 
        NUM = MULT_ALL
        DEN = MULT_ALL + (1 - pred_model1_sample) * (1 - pred_model2_sample) * (1 - util.A_PRIOR_PROB)
        pred_sample = NUM / DEN
        preds_sample.append(pred_sample)
    return np.array(preds_sample)


               
               
