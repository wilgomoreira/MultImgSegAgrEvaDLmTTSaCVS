from sweep_up import SweepUp
from refine_preds import RefinePreds
from metrics import Metrics
from print.print_all import PrintAll
from chart import Chart

def main():
    print("RUNNING....")
    all_images = SweepUp.get_all_images()
    all = RefinePreds.apply(all_images=all_images)
    #Chart.histogram(preds_crf, nos_crfs)
    #all_metrics = Metrics.for_all_images(all_images=all_images)
    #PrintAll.files(all_metrics) 
    print("FINISH!!")
        
if __name__ == "__main__":
    main()

