import torch
import numpy as np
from netcal.scaling import TemperatureScaling
from netcal.metrics import ECE
import util


class Calibration:
    def __init__(self, model, dataloader, args, param):
        if util.CALIBRATION:
            do_tempScalling(model=model, dataloader=dataloader, args=args, param=param)


# Support Functions -----------------------------------------------------------------
def do_tempScalling(model, dataloader, args, param):
    train_dataloader = dataloader.train_dataloader
    test_dataloader = dataloader.test_dataloader
    
    # Obter logits e rótulos no conjunto de treinamento
    train_logits, train_labels = get_logits_and_labels(model, train_dataloader, args, param.spectrum_name)
    train_logits = train_logits.flatten()
    train_labels = train_labels.flatten()
    
    # Ajustar o Temperature Scaling usando os logits do conjunto de treinamento
    ts = TemperatureScaling()
    train_probs = sigmoid(train_logits)
    ts.fit(train_probs, train_labels)

    # Obter logits e rótulos no conjunto de teste
    test_logits, test_labels = get_logits_and_labels(model, test_dataloader, args, param.spectrum_name)
    test_logits = test_logits.flatten()
    test_labels = test_labels.flatten()

    # Calibrar os logits do conjunto de teste
    test_probs = sigmoid(test_logits)
    calibrated_test_probs = ts.transform(test_probs)

    # Calcular métricas de calibração
    ece = ECE(util.N_BINS)  # Assegurando que n_bins é passado corretamente
    uncalibrated_score = ece.measure(test_probs, test_labels)
    calibrated_score = ece.measure(calibrated_test_probs, test_labels)
    
    evaluate_perfMetrics(calibrated_test_probs, test_probs, test_labels)
    
    with open(util.OUT_RESULTS, 'a') as f:
        f.write("----------------------------------------------------------\n")
        f.write(f"SPECTRUM: {param.spectrum_name} - MODEL: {param.model_name} - DATASET: {param.dataset_name}\n")
        f.write(f"Original ECE: {uncalibrated_score}\n")
        f.write(f"Calibrated ECE: {calibrated_score}\n")
        f.write(f"DIFFERENCE ECE: {calibrated_score - uncalibrated_score}\n")

def get_logits_and_labels(model, dataloader, args, spec_name):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for data in dataloader:
            rgb, ndvi, gndvi, mask, id = data
            rgb, ndvi, gndvi, mask = util.FUNC.SHAPE_GPU.evaluate(args=args, rgb=rgb, ndvi=ndvi, 
                                                                  gndvi=gndvi, mask=mask)
            input_mod = util.FUNC.DECISION.choose_input_mode(spec_name=spec_name, rgb=rgb, ndvi=ndvi, gndvi=gndvi)
            t0, t1, output = util.FUNC.DECISION.time_for_model(model=model, input_mod=input_mod)
            
            # Assuming mask contains per-example labels, not per-pixel
            logits = output.cpu().numpy()  # Obtendo os logits diretamente
            # Ensure logits are 2D
            logits = logits.reshape(logits.shape[0], -1)
            all_logits.append(logits)
            
            labels = mask.cpu().numpy()
            labels = labels.reshape(labels.shape[0], -1)
            all_labels.append(labels)

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    
    return all_logits, all_labels

def sigmoid(vector):
    v_torch = torch.from_numpy(vector)
    v_prob = torch.sigmoid(v_torch)
    return v_prob.numpy()

def evaluate_perfMetrics(calibrated_test_probs, test_probs, test_labels):
    calibrated_test_probs = torch.from_numpy(calibrated_test_probs)
    test_probs = torch.from_numpy(test_probs)
    test_labels = torch.from_numpy(test_labels)
    
    calibrated_preds = util.FUNC.DECISION.take_a_one(probs=calibrated_test_probs)
    test_preds = util.FUNC.DECISION.take_a_one(probs=test_probs)
    
    calib_metrs = util.FUNC.METRIC.compute_performance_metrics(preds=calibrated_preds, mask=test_labels)
    metrs = util.FUNC.METRIC.compute_performance_metrics(preds=test_preds, mask=test_labels)
    
    _, _, calib_iou, _, _ = calib_metrs
    _, _, iou, _, _ = metrs
    
    print(f'CALIBRATED IOU: {calib_iou} - IO: {iou}')
    print("TEST")