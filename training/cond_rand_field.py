import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import numpy as np
import util
from pydensecrf.utils import create_pairwise_bilateral
import matplotlib.colors as mcolors

class CondRandField:
    
    @staticmethod
    def evaluate(preds, inputs):
        
        preds_crf = []
        for pred, input_i in zip(preds, inputs):
            value_crf = _CRF_samples(preds=pred, inputs=input_i)
            value_crf = np.array(value_crf)
            pred.value = value_crf
            preds_crf.append(pred)

        return preds_crf
      
def _CRF_samples(preds, inputs):
    
    if inputs.spectrum == util.NDVI or inputs.spectrum == util.GNDVI:
        inputs = normalise_enphasise_colours(inputs.value)
        inputs = np.squeeze(inputs)
    else:
        inputs = normalise_enphasise_colours(inputs.value)
    
    preds = preds.value
    
    if util.CALIBRATION == False:
        preds = util.sigmoid(preds)
    
    preds_samples = []
    for pred_sample, input_sample in zip(preds, inputs):
        back_preds = 1 - pred_sample
        fore_preds = pred_sample
        joined_preds = np.vstack((back_preds, fore_preds)).reshape((2, util.IMG_HEIGHT, 
                                                                    util.IMG_WIDTH))
        
        unary = unary_from_softmax(joined_preds)
        d = dcrf.DenseCRF2D(joined_preds.shape[2], joined_preds.shape[1], 2)
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=util.PAIRWISEGAUS_SXY, compat=util.PAIRWISEGAUS_COMPAT)
        
        if input_sample.dtype != np.uint8:
            input_sample = (input_sample * 255).astype(np.uint8)
        else:
            input_sample = input_sample
            
        if input_sample.shape != (240, 240, 3):
            input_sample = np.transpose(input_sample, (1, 2, 0))
            
        if not input_sample.flags['C_CONTIGUOUS']:
            input_sample = np.ascontiguousarray(input_sample)
        
        pairwise_bilateral = create_pairwise_bilateral(sdims=(util.PAIRBILATERAL_SXY, util.PAIRBILATERAL_SXY), 
                                               schan=(util.PAIRBILATERAL_SRGB, util.PAIRBILATERAL_SRGB, util.PAIRBILATERAL_SRGB),
                                               img=input_sample, chdim=2)
    
        d.addPairwiseEnergy(pairwise_bilateral, compat=util.PAIRBILATERAL_COMPAT)
        
        Q = d.inference(util.INFERENCE)
        
        # Output in likehood format
        Q_np = np.array(Q)
        Q_reshaped = Q_np.reshape((-1, util.IMG_HEIGHT, util.IMG_WIDTH))
        probs = Q_reshaped[1, :, :]
        new_prob = np.expand_dims(probs, axis=0)
        
        preds_samples.append(new_prob)
        
    return preds_samples
               
               
def normalise_enphasise_colours(images):
    processed_images = []
    
    for image in images:
        # Para imagens RGB
        if image.shape[0] == 3:
            processed_image = image
        else:
            # Para NDVI/GNDVI, continua com a normalização e aplicação da paleta de cores
            image_normalized = (image + 1) / 2  # Normaliza para o intervalo [0, 1]
            
            colors = [(165/255, 42/255, 42/255), (255/255, 255/255, 224/255), (34/255, 139/255, 34/255)]
            n_bins = [0, 0.5, 1]
            cmap_name = 'my_list'
            cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, list(zip(n_bins, colors)))
            
            image_colored = cm(image_normalized)  # Aplica a paleta de cores
            processed_image = image_colored[..., :3]  # Remove o canal alpha
            processed_image = (processed_image * 255).astype(np.uint8)  # Converte para uint8 se necessário
            
        processed_images.append(processed_image)
    
    return np.array(processed_images)




