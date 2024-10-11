from model_segnet import SegNet
import network
from dataclasses import dataclass
import util

@dataclass
class SelectModel:
    model: any
    
    @classmethod
    def choose_one(clc, model_name, spectrum_name=util.RGB):
        _recoginzed_model_spectrum(model_name, spectrum_name)
        input_channels = _number_channels(spectrum_name)
        model = _call_model(model_name, input_channels)
                
        return clc(model=model)
    
def _recoginzed_model_spectrum(model_name, spectrum_name):   
    assert model_name.lower() in util.MODELS, "Model not recognized"  
    assert spectrum_name.upper() in util.SPECTRUMS, "Spectrum not recognized" 

def _number_channels(spectrum_name):
    
    match spectrum_name:   
        
        case util.RGB:
            input_channels = util.INPUT_RGB_CHAN      
        
        case util.NDVI:
            input_channels = util.INPUT_NDVI_CHAN     
            
        case util.GNDVI:
            input_channels = util.INPUT_GNDVI_CHAN
        
        case util.EARLY_FUSION:
            input_channels = util.INPUT_RGB_CHAN + util.INPUT_NDVI_CHAN + util.INPUT_GNDVI_CHAN
    
    return input_channels

def _call_model(model_name, input_channels):
    model = None
                
    match model_name:
        
        case util.SEGNET:
            model = SegNet(num_classes=util.NUM_CLASSES, n_init_features=input_channels)
        
        case util.DEEPLAB:
            DEEPLAB_VERSION = util.DEEPLAB_VERSION
            OUTPUT_STRID = util.DEEPLAB_OUTPUT_STRIDE
            model = network.modeling.__dict__[DEEPLAB_VERSION](in_channels=input_channels,
                                                                num_classes=util.NUM_CLASSES, 
                                                                output_stride=OUTPUT_STRID)   
    return model
                
                
                
            
                
        
    
        