import torch
import torch.nn as nn
import torch.nn.functional as F
from modelo import PositionalEncoding as pose

class Modelo(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_classes, dropout=0.05):
        super(Modelo, self).__init__()
        self.positional_encoding = pose(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.output_layer(output)
        return F.log_softmax(output, dim=1)

# Ejemplo de uso
if __name__ == "__main__":
    num_classes = 10 # Por ajustar
    modelo = Modelo(d_model=200, nhead=4, num_layers=3, num_classes=num_classes)
    src = torch.rand(32, 10, 100) # Suponiendo un batch_size=32, seq_length=10, d_model=100
    out = modelo(src)
    print(out) # salida del modelo
