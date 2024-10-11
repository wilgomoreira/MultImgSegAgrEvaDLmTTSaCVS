import numpy as np
import matplotlib.pyplot as plt

class Chart:
    @staticmethod
    def histogram(preds_crf, preds_no_crf):
        
        def min_max_scale(data):
            min_val = np.min(data)
            max_val = np.max(data)
            scaled_data = (data - min_val) / (max_val - min_val)
            return scaled_data
        
        def calculate_max_normalized_histogram(data, bins):
            counts, bin_edges = np.histogram(data, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            max_count = np.max(counts)
            normalized_counts = counts / max_count
            return bin_centers, normalized_counts
        
        # Processamento para imagens não refinadas
        all_images_flat = np.concatenate([img.flatten() for img in preds_no_crf])
        all_images_flat_scaled = min_max_scale(all_images_flat)
        
        bin_centers, normalized_counts = calculate_max_normalized_histogram(all_images_flat_scaled, bins=50)
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
        axes[0].plot(bin_centers, normalized_counts, drawstyle='steps-mid', label='Preds', color='black')
        
        axes[0].set_title('Histogram of Predictions')
        axes[0].set_xlabel('Scaled Value')
        axes[0].set_ylabel('Normalized Density')
        axes[0].set_ylim(0, 1.1)  # Definir limite do eixo y de 0 a 1
        axes[0].legend()
        axes[0].grid(True)

        # Processamento para previsões refinadas (com CRF)
        preds_refined_flat = np.concatenate([pred.flatten() for pred in preds_crf])
        preds_refined_flat_scaled = min_max_scale(preds_refined_flat)
        
        bin_centers, normalized_counts = calculate_max_normalized_histogram(preds_refined_flat_scaled, bins=50)
        axes[1].plot(bin_centers, normalized_counts, drawstyle='steps-mid', label='Preds with CRF', color='black')

        axes[1].set_title('Histogram of Predictions applied CRF')
        axes[1].set_xlabel('Scaled Value')
        axes[1].set_ylabel('Normalized Density')
        axes[1].set_ylim(0, 1.1)  # Definir limite do eixo y de 0 a 1
        axes[1].legend()
        axes[1].grid(True)

        # Ajustar o layout e salvar
        plt.tight_layout()
        plt.savefig('/home/wilgo/Downloads/MISAgro1.0/post_training/print/histograms_combined.png')
        plt.close()