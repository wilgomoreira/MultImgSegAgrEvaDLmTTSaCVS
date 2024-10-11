import util
from calibration import Calibration
from cond_rand_field import CondRandField
from bayes import Bayes

class RefinePreds:
    
    @staticmethod
    def apply(all_images):
        inputs, masks, preds = all_images
        
        if util.CALIBRATION:
            preds = _calibration(mask=masks, preds=preds)
        
        if util.CONDICIONAL_RANDOM_FIELD:
            preds = CondRandField.evaluate(inputs=inputs, preds=preds, masks=masks)
            
        if util.BAYES:
            preds = _apply_bayes(preds=preds)
    
        #return inputs, masks, preds
        return preds
    
def _calibration(mask, preds):
    
    preds_calib = Calibration.evaluate(make_calib=util.CALIB_MET,
                                       raw_preds=preds, masks=mask)
    return preds_calib


def _apply_bayes(preds):
    
    preds_bayes = Bayes.combine_likehoods_models(preds=preds)
    return preds_bayes