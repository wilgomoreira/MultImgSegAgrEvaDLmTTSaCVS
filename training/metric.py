import torch
from dataclasses import dataclass
import os
import pickle
import util

@dataclass
class Metric:
    time_duration: list
    accuracies: list
    f1_scores: list
    ious: list
    precisions: list
    recalls: list 
    
    @classmethod
    def evaluate(clc, args, save_pred_mask_dir, spec_name, dataloader, model):
        time_duration, accuracies, f1_scores, recalls, precisions, ious = [], [], [], [], [], []
        
        if util.DATALOADER_RESULTS == util.TEST_DATALOADER or util.TRAIN_MODEL == True:
            chosen_dataloader = dataloader.test_dataloader
        else:
            chosen_dataloader = dataloader.train_dataloader
    
        with torch.no_grad():
            for it, data in enumerate(chosen_dataloader):
                rgb, ndvi, gndvi, mask, id = data   
                rgb, ndvi, gndvi, mask = util.FUNC.SHAPE_GPU.evaluate(args=args, rgb=rgb, ndvi=ndvi, 
                                                                      gndvi=gndvi, mask=mask)
                    
                input_mod = util.FUNC.DECISION.choose_input_mode(spec_name=spec_name, rgb=rgb, ndvi=ndvi, gndvi=gndvi)
                t0, t1, output = util.FUNC.DECISION.time_for_model(model=model, input_mod=input_mod)
                probs = torch.sigmoid(output)
                preds = util.FUNC.DECISION.take_a_one(probs=probs)
                metrs = util.FUNC.METRIC.compute_performance_metrics(preds=preds, mask=mask)
                
                acc, f1_score, iou, precision, recall = metrs
                time_duration.append(t1-t0)
                accuracies.append(acc)
                f1_scores.append(f1_score)
                ious.append(iou)
                precisions.append(precision)
                recalls.append(recall)
                
                input_mod = input_mod.detach().cpu().numpy()
                mask = mask.cpu().detach().numpy()
                probs = probs.cpu().detach().numpy()
                
                _for_post_training(input_mod=input_mod, mask=mask, probs=output, id=id, 
                                   save_pred_mask_dir=save_pred_mask_dir)
        
        return clc(time_duration=time_duration, accuracies=accuracies, 
                   f1_scores=f1_scores, ious=ious, precisions=precisions, recalls=recalls)

# Support Functions -----------------------------------------------------------------
def _for_post_training(input_mod, mask, probs, id, save_pred_mask_dir):
            
    for i, (input, mask, pred, label) in enumerate(zip(input_mod, mask, probs, id)):
        with open(os.path.join(save_pred_mask_dir, label + '.pickle'), 'wb') as handle:
            pickle.dump({'input':input,'mask':mask,'pred':pred}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        