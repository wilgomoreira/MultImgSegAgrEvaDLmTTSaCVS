import util

class Print():
    
    @staticmethod
    def screen_and_file(param, metr):
        time_duration = metr.time_duration
        metrcs = (metr.accuracies, metr.f1_scores, metr.ious, metr.precisions, metr.recalls)
        mean_metr_perc = util.FUNC.METRIC.mean__metrics_perc(metrs=metrcs, time_duration=time_duration)
        mean_duration, mean_acc_perc, mean_precis_perc, mean_recall_perc, mean_f1_scor_perc, mean_iou_perc = mean_metr_perc
        
        content =  (f"| - SAVED MODEL - {param.spectrum_name} - Model {param.model_name} Dataset {param.dataset_name} " +
                    f"Duration(s): {mean_duration} Acc: {mean_acc_perc}%. Prec: {mean_precis_perc}%." +
                    f"Recall: {mean_recall_perc}%. F1: {mean_f1_scor_perc}%. IoU: {mean_iou_perc}%")   
        print(content)

        with open(util.SAVE.OVERAL_PERFM, 'a') as file:
            file.write(content)
            file.write('\n')   
        
        
    

         