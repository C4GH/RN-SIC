import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
from modelo import PositionalEncoding as pose

class Modelo(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_classes, dropout=0.05):
        super(Modelo, self).__init__()
        self.positional_encoding = pose.PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.output_layer(output)
        return F.log_softmax(output, dim=1)

# Configuración del modelo
d_model = 100 # Debe ser igual al embedding
nhead = 4
num_layers = 3
num_classes = 10 # Asignado arbitrariamente
dropout = 0.05

# Instanciación del demolo
modelo = Modelo(d_model=d_model, nhead=nhead, num_layers=num_layers, num_classes=num_classes, dropout=dropout)

# Creación de una entrada de ejemplo y medición de la dimensionalidad
input_tensor = torch.rand(32,10, d_model) # batch_size=31, sequence_length=10, embedding_dim=d_model
print(f"Dimensionalidad de la entrada: {input_tensor.shape}")

# Pasada de la entrada por el modelo
output_tensor = modelo(input_tensor)
print(f"Dimensionalidad de la salida: {output_tensor.shape}")

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

"""
Dimensionalidad de la salida: torch.Size([32, 10])
Total de parámetros entrenables en el modelo: 1358654
Memoria aproximada de los parámetros: 5.182853698730469 MB
Uso de memoria del proceso: 195.62109375 MB
"""