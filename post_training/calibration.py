from sklearn.isotonic import IsotonicRegression
import numpy as np
import util
from scipy.optimize import minimize_scalar

class Calibration:
    
    @staticmethod
    def evaluate(make_calib, raw_preds, masks):
        
        
        
        match make_calib:
            case util.CALIB_ISO_REGR:
                preds_calib = [iso_regression(util.sigmoid(logits.value), mask.value) for logits, mask in zip(raw_preds, masks)]
            case util.CALIB_TEMPERATURE:
                preds_calib = [temperature_scaling(logits.value, mask.value) for logits, mask in zip(raw_preds, masks)]
        
        for logits, value_calib in zip(raw_preds, preds_calib):
            logits.value = np.array(value_calib)
        
        return raw_preds
        
def iso_regression(logits, masks):
    logits_calib = []
    
    for logits_sample, mask_sample in zip(logits, masks):
        ir = IsotonicRegression(out_of_bounds='clip')
        #transform to vector
        logits_sample_flat = logits_sample.flatten()
        mask_sample_flat = mask_sample.flatten()
        #calibration
        prob_calib = ir.fit_transform(logits_sample_flat, mask_sample_flat)
        
        #return to the shape
        prob_calib_matrix = prob_calib.reshape(1, util.IMG_HEIGHT, util.IMG_WIDTH)
        logits_calib.append(prob_calib_matrix)
        
    return logits_calib

def temperature_scaling(logits, masks):
    logits_calib = []
    
    for logits_sample, mask_sample in zip(logits, masks):
        result = minimize_scalar(_nll_loss, bounds=(0.1, 10.0), method='bounded', args=(logits_sample, mask_sample))
        temperature_opt = result.x
        
        logits_calib_samples = util.sigmoid(logits_sample / temperature_opt)
        
        logits_calib_samples_reshaped = logits_calib_samples.reshape(1, util.IMG_HEIGHT, util.IMG_WIDTH)
        logits_calib.append(logits_calib_samples_reshaped)

    return logits_calib
            
def _nll_loss(temperature, logits, masks):
    ZERO = 1e-8
    ONE_MINUS_ZERO = 1 - ZERO

    logits_adjusted = logits / max(temperature, ZERO)
    probs_temp = util.sigmoid(logits_adjusted)
    probs_temp_clipped = np.clip(probs_temp, ZERO, ONE_MINUS_ZERO)

    loss_pos = -np.log(np.clip(probs_temp_clipped[masks == 1], ZERO, ONE_MINUS_ZERO))
    loss_neg = -np.log(np.clip(1 - probs_temp_clipped[masks == 0], ZERO, ONE_MINUS_ZERO))
    loss = np.mean(np.concatenate([loss_pos, loss_neg]))

    return loss


