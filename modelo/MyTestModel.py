import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import psutil
import sys
from modelo import PositionalEncoding

# Suponiendo que PositionalEncoding y Modelo ya están definidos como antes

# Carga los embeddings pre-entrenados
embeddings_path = 'C:\\Users\\afloresre\\Documents\\Cagh\\Red\\salida\\embs_npa.npy'
embeddings = np.load(embeddings_path)
embeddings_tensor = torch.tensor(embeddings, dtype=torch.float)

# Configuración e instancia del modelo
d_model = embeddings.shape[1]
nhead = 4
num_layers = 3
num_classes = 10
dropout = 0.05

modelo = Modelo(d_model, nhead, num_layers, num_classes, dropout, embeddings_tensor)

# Índices de prueba
input_indices = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

# Cálculo del uso de memoria de los parámetros
total_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
total_bytes = total_params * 4  # Asumiendo float32
total_megabytes = total_bytes / (1024 ** 2)
print(f"Total de parámetros entrenables en el modelo: {total_params}")
print(f"Memoria aproximada de los parámetros: {total_megabytes} MB")

# Medición del uso de memoria del proceso utilizando psutil
process = psutil.Process()
memory_info = process.memory_info()
memory_usage_mb = memory_info.rss / (1024 ** 2)  # Convertir bytes a MB
print(f"Uso de memoria del proceso: {memory_usage_mb} MB")

# Uso de memoria del modelo (aproximación)
# Nota: sys.getsizeof no es muy preciso para objetos complejos como los modelos de PyTorch,
# pero te da una idea general.
memory_usage_modelo_bytes = sys.getsizeof(modelo)
print(f"Uso de memoria del modelo (aproximación): {memory_usage_modelo_bytes} bytes")

# Realiza una operación
output = modelo(input_indices)
print("Salida del modelo:", output)