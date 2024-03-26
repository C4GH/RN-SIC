import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Cálculo del total de parámetros entrenables en el modelo
total_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
print(f"Total de parámetros entrenables en el modelo: {total_params}")

# Total fue de 1358654