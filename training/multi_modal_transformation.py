import torch
import cv2
import random
from torchvision.transforms import functional as F
from torchvision import transforms as T

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb, mask):
        for t in self.transforms:
            rgb, mask = t(rgb, mask)
        
        return rgb, mask
    
class ToTensor:
    def __init__(self):
        self.tensor = T.ToTensor()

    def __call__(self, rgb, mask):
        if not torch.is_tensor(rgb):
            mask = cv2.convertScaleAbs(mask)  
            rgb = self.tensor(rgb,)
            mask = self.tensor(mask)       
           
            mask = mask.type(torch.float32)
            rgb  = rgb.type(torch.float32)
        
        return rgb, mask
            
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, rgb, mask):
        rgb = F.hflip(rgb)
        mask = F.hflip(mask)
        
        return rgb, mask
       
class AdjustSaturation:
    def __init__(self, saturation_value=4):
        self.max_sat_value = saturation_value
    
    def __call__(self, rgb, mask):
        sat = random.random() * self.max_sat_value
        rgb = F.adjust_saturation(rgb, sat)
        
        return rgb, mask

class AdjustBrightness:
    def __init__(self, brightness_factor = 3):
        self.brightness_factor = brightness_factor
    
    def __call__(self, rgb, mask):
        value = random.random() * self.brightness_factor
        rgb = F.adjust_brightness(rgb, value)
        
        return rgb, mask