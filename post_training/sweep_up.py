from spectrum import Spectrum
import util

class SweepUp:
    
    @staticmethod
    def get_all_images():
        all_inputs, all_masks, all_preds = [], [], []

        for model in util.MODELS:
            for database in util.DATABASES:
                inputs, masks, preds = Spectrum.inputs_masks_preds(model=model, database=database)
                all_inputs.extend(inputs)
                all_masks.extend(masks)
                all_preds.extend(preds)
       
        return all_inputs, all_masks, all_preds
    
