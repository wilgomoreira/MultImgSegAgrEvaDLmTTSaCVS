from matplotlib import pyplot as plt
import torch
import random
import numpy as np
from operator import attrgetter
import util

class PlotBestResults:
    
    @staticmethod
    def do(objs):
        fig_best = plt.figure()
        fig_best.set_size_inches(util.PLOT.SUBPLOT_SIZE_BEST)
        fig_best.suptitle(util.PLOT.SUBTITLE_BEST) 
        best_objects = _get_best_results(objs)
        _show_best(best_objects)
        
def _get_best_results(objs): 
    sorted_objects = _sort_metrics(objs)
    best_objects = []
    
    for i in range(util.PLOT.N_RESULTS):
        best_objects.append(sorted_objects[i]) 
    return best_objects
       
def _sort_metrics(objs):
    item_sort = f"mean_{util.TABLE.ITEM_SORT}"
    sorted_objects = sorted(objs, key=attrgetter(item_sort), reverse=util.PLOT.REVERSE)
    return sorted_objects

def _show_best(best_objects):
    n_rows = len(best_objects)
    n_columns = util.PLOT.N_COLUMNS
    
    index_img = 1
    
    for best_obj in best_objects:
        _images_in_columns(n_rows, index_img, best_obj)
        index_img += n_columns
        
    plt.savefig(util.PLOT.BEST_RESULT_PATH)
          
def _images_in_columns(n_row, index_img, best_obj):
    column_objs = (best_obj.inputs, best_obj.masks, best_obj.preds)
    chosen_img = _chosen_image(best_obj)
    num_objetcs = len(column_objs)
    
    for i, obj in enumerate(column_objs):
        subp_title = _subplots_title(obj)
        bin_image = _prepare_image(i, obj, chosen_img)
        
        plt.subplot(n_row, num_objetcs, index_img)
        plt.title(subp_title)
        plt.imshow(bin_image)
        plt.xticks([]), plt.yticks([])
        index_img += 1
        
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

        case util.PLOT.CRIT_DEFINE:
            return util.PLOT.IMG_DEFINED
            
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
    
def _prepare_image(i, raw_object, num_random):
    raw_images = raw_object.value
    raw_image = raw_images[num_random]
        
    inv_image = torch.from_numpy(raw_image)
    image = torch.permute(inv_image, util.PLOT.PERMUT_IMG)
    
    if not raw_object.name == util.PLOT.NOT_DECISION:
        image = util.thresholding(image)
    return image
