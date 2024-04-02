"""
Implementación correcta con batch al inicio
"""

from modelo import Modelo4 as M
import torch
# import resource
import psutil
import time
import sys
import torch.nn.functional as F

batch_size = 50
max_tokens = 200
num_layers = 3
d_model = 512
nheads = 4

src_emb = torch.rand(batch_size, max_tokens, d_model)
print(f'src_emb shape: {src_emb.shape}')

modelo = M.Modelo(d_model, nheads, num_layers, max_tokens, 3)

tam_modelo = sys.getsizeof(modelo)
# recursos = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# Implementación de psutil para calculo de uso de memoria en Windows
process = psutil.Process()
recursos = process.memory_info().rss
total_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)

print('forward')
out = modelo.forward(src_emb)

print(f'out shape: {out.shape}')
print(out)
print(f'Uso de memoria (en bytes): {recursos}')
print(f'Tamaño del modelo (en bytes): {tam_modelo}')
print(f'Número total de parámetros entrenables: {total_params}')