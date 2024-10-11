import numpy as np

def transform_to_two_channel(matrix):
    num_samples, _, altura, largura = matrix.shape
    new_matrix = np.zeros((num_samples, 2, altura, largura), dtype=int)

    # Classe negativa (0) no primeiro canal
    new_matrix[:, 0, :, :] = (matrix[:, 0, :, :] == 0).astype(int)
    
    # Classe positiva (1) no segundo canal
    new_matrix[:, 1, :, :] = (matrix[:, 0, :, :] == 1).astype(int)

    return new_matrix

# Exemplo de uso
num_samples = 10
altura = 240
largura = 240
original_matrix = np.random.randint(0, 2, (num_samples, 1, altura, largura))

transformed_matrix = transform_to_two_channel(original_matrix)
print(f'Original matrix shape: {original_matrix.shape}')
print(f'Transformed matrix shape: {transformed_matrix.shape}')