import numpy as np
from scipy.optimize import minimize
import torch
import util
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from sklearn.metrics import precision_recall_fscore_support


class CalibrationCRF:
    def __init__(self, model, dataloader, args, param):
        if util.USE_CALIBRATION:
            calibrated_test_logits, test_logits, test_labels = self.do_temp_scaling(
                                                    model=model, dataloader=dataloader, 
                                                    args=args, param=param)
        if util.USE_CRF:
            preds = self.apply_crf_to_samples(calibrated_test_logits)
        
        f1_score = calculate_f1_scores(softmax(test_logits), test_labels)
        print(f"F1_SCORE: {f1_score}")
        

    def do_temp_scaling(self, model, dataloader, args, param):
        train_dataloader = dataloader.train_dataloader
        test_dataloader = dataloader.test_dataloader

        # Obter logits e rótulos no conjunto de treinamento
        train_logits, train_labels = self.get_logits_and_labels(model, train_dataloader, args, param.spectrum_name)

        # Encontrar a temperatura ótima
        temperature = self.find_optimal_temperature(train_logits, train_labels)

        # Aplicar a temperatura aos logits do conjunto de teste
        test_logits, test_labels = self.get_logits_and_labels(model, test_dataloader, args, param.spectrum_name)
        calibrated_test_logits = test_logits / temperature
        
        #calibrated_test_probs = torch.sigmoid(torch.from_numpy(calibrated_test_logits)).numpy()
        
        return calibrated_test_logits, test_logits, test_labels

        # Avaliar usando logits calibrados
        #self.evaluate_metrics_ece(test_logits, test_labels, param)
        #self.evaluate_metrics_ece(calibrated_test_logits, test_labels, param, temperature)

    def get_logits_and_labels(self, model, dataloader, args, spec_name):
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

    def find_optimal_temperature(self, logits, labels):
        def objective(temperature):
            # Calcular a perda logística nos logits ajustados pela temperatura
            scaled_logits = logits / temperature
            loss = torch.nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(scaled_logits), torch.from_numpy(labels))
            return loss.item()

        # Otimizar a temperatura
        result = minimize(objective, x0=1.0, bounds=[(0.01, 5.0)])
        return result.x[0]
    
    def evaluate_metrics_ece(self, logits, labels, param, temperature=None):
        if temperature is not None:
            # Aplicar temperatura aos logits para calibração
            logits = logits / temperature
        
        probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()

        # Número de bins para o cálculo do ECE
        n_bins = 10
        bin_limits = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_limits[:-1]
        bin_uppers = bin_limits[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Encontrar as amostras neste bin
            in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)
            bin_probs = probabilities[in_bin]
            bin_labels = labels[in_bin]

            if len(bin_probs) > 0:
                # Calcular a precisão e a confiança para este bin
                accuracy = np.mean(bin_labels == (bin_probs > 0.5))
                confidence = np.mean(bin_probs)

                # Calcular a contribuição deste bin para o ECE
                bin_weight = len(bin_probs) / len(probabilities)
                ece += np.abs(accuracy - confidence) * bin_weight

        print(f"ECE for {'calibrated' if temperature is not None else 'uncalibrated'} logits: {ece}")
        return ece
    
    def apply_crf_to_samples(self, logits):
        num_classes = 2
        height = 240
        width = 240
        
        reformatted_logits = reformat_logits(logits, num_classes, height, width) 
        predicted_segmentations = []
        
        for logits in reformatted_logits:
            # Converter logits para probabilidades (usando softmax)
            probabilities = softmax(logits)  # Defina a função softmax conforme necessário
            # Configurar e aplicar CRF (ajuste conforme necessário)
            d = dcrf.DenseCRF2D(width, height, logits.shape[0])
            U = unary_from_softmax(probabilities)
            d.setUnaryEnergy(U.copy(order='C'))
            d.addPairwiseGaussian(sxy=3, compat=3)
            Q = d.inference(5)
            map_soln = np.argmax(Q, axis=0).reshape((height, width))
            predicted_segmentations.append(map_soln)
        return np.array(predicted_segmentations)
    
def softmax(logits):
    e_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
    probabilities = e_logits / np.sum(e_logits, axis=0, keepdims=True)
    return probabilities

def reformat_logits(logits, num_classes, height, width):
    # Primeiro, reshape para (numero_de_amostras, altura, largura, numero_de_classes)
    reshaped_logits = logits.reshape((-1, height, width, num_classes))
    # Transpor para (numero_de_amostras, numero_de_classes, altura, largura)
    reformatted_logits = reshaped_logits.transpose(0, 3, 1, 2)
    return reformatted_logits

def calculate_f1_scores(predictions, labels):
    f1_scores = []
    for pred, label in zip(predictions, labels):
        # Achata as predições e labels para 1D
        pred_flat = pred.flatten()
        label_flat = label.flatten()
        
        pred_flat_binary = (pred_flat > 0.5).astype(int)
        
        # Calcula precision, recall, e F1 score para a amostra atual
        precision, recall, f1, _ = precision_recall_fscore_support(label_flat, pred_flat_binary, average='binary', zero_division=0)
        f1_scores.append(f1)
    
    return np.mean(f1_scores)