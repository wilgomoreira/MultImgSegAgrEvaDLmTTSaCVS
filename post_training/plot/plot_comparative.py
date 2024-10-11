from matplotlib import pyplot as plt
import torch
import random
import numpy as np
import util

class PlotComparative:
    
    @staticmethod
    def do(objs):
        fig_comp = plt.figure()
        fig_comp.set_size_inches(util.PLOT.SUBPLOT_SIZE_COMP)
        fig_comp.suptitle(util.PLOT.SUBTITLE_COMP) 
        imgs = _define_comparative(objs)
        _show_comparative(imgs)
        
def _define_comparative(objs):
    for obj in objs:
        if (obj.inputs.name == util.INPUT and obj.inputs.model == util.PLOT.MODEL1_COMP and 
            obj.inputs.database == util.PLOT.DATA1_COMP and 
            obj.inputs.spectrum == util.PLOT.DEFAULT_SPEC_COMP):
            input_obj = obj
        
        if (obj.masks.name == util.MASK and obj.masks.model == util.PLOT.MODEL1_COMP and 
            obj.masks.database == util.PLOT.DATA1_COMP and 
            obj.masks.spectrum == util.PLOT.DEFAULT_SPEC_COMP): 
            mask_obj = obj
        
        if (obj.name == util.PLOT.PRED1NAME_COMP and obj.model == util.PLOT.MODEL1_COMP and 
            obj.database == util.PLOT.DATA1_COMP and 
            obj.spectrum == util.PLOT.SPECTRUM1_COMP): 
            lfusion_wm = obj
        
        if (obj.name == util.PLOT.PRED2NAME_COMP and obj.model == util.PLOT.MODEL2_COMP and 
            obj.database == util.PLOT.DATA2_COMP and 
            obj.spectrum == util.PLOT.SPECTRUM2_COMP): 
            lfusion_m = obj
        
        if (obj.name == util.PLOT.PRED3NAME_COMP and obj.model == util.PLOT.MODEL3_COMP and 
            obj.database == util.PLOT.DATA3_COMP and 
            obj.spectrum == util.PLOT.SPECTRUM3_COMP): 
            efusion = obj 
    
    return input_obj, mask_obj, lfusion_wm, lfusion_m, efusion
    
def _show_comparative(imgs):
    input_obj, mask_obj, lfusion_wm, lfusion_m, efusion = imgs 
    chosen_img = _chosen_image(lfusion_wm)
    
    index_img = 1
    imgs = input_obj.inputs, mask_obj.masks, lfusion_wm.preds, lfusion_m.preds, efusion.preds
    
    for img in imgs:
        subp_title = _subplots_title(img)
        bin_image = _prepare_image(img, chosen_img)
        
        plt.subplot(1, 5, index_img)
        plt.title(subp_title)
        plt.imshow(bin_image)
        plt.xticks([]), plt.yticks([])
        index_img += 1
        
    plt.savefig(util.PLOT.COMPARATIVE_PATH)
                        
def _chosen_image(best_obj):
    preds = best_obj.preds.value
    list_f1s = best_obj.f1s
    chosen_img = util.PLOT.CHOSEN_IMG
    
    match chosen_img:
        
        case util.PLOT.CRIT_RANDOM:
            num_images = len(preds)
            num_random = random.randint(0, num_images)
            return num_random 
        
        case util.PLOT.CRIT_MAX:
            index_max_value = np.argmax(list_f1s)
            return index_max_value
        
        case util.PLOT.CRIT_MIN:
            index_min_value = np.argmin(list_f1s)
            return index_min_value
    
        case util.PLOT.CRIT_SIMILAR_MEAN:
            mean_f1s = best_obj.mean_f1s
            tol = util.PLOT.TOL_SIMILAR_MEAN
        
            index_similar_mean = next(i for i, _ in enumerate(list_f1s) if np.isclose(_, mean_f1s, tol))
            return index_similar_mean
            
def _subplots_title(raw_object):
    name_image = raw_object.name
    model_image = raw_object.model
    database_image = raw_object.database
    spectrum_image = raw_object.spectrum
    
    if name_image == util.MASK:
        subp_title = f"{database_image.upper()}: {model_image.upper()} - {name_image}" 
    else:
        subp_title = f"{database_image.upper()}: {model_image.upper()} - {spectrum_image.upper()}"   
    return subp_title
     
def _prepare_image(raw_object, num_random):
    raw_images = raw_object.value
    raw_image = raw_images[num_random]
        
    inv_image = torch.from_numpy(raw_image)
    image = torch.permute(inv_image, util.PLOT.PERMUT_IMG)
    
    if not raw_object.name == util.PLOT.NOT_DECISION:
        image = util.thresholding(image)
    return image
