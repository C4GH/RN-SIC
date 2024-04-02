import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from modelo import PositionalEncoding as pose

class Modelo(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_tokens, num_classes, dropout=0.05):
        super(Modelo, self).__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens

        self.positional_encoding = pose.PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model * num_tokens, num_classes)

    def forward(self, emb1):
        batch_size = emb1.shape[0]
        emb1_pe = self.positional_encoding(emb1)
        sal_en = self.transformer_encoder(emb1_pe)
        sal_en_aplanado = sal_en.view([batch_size, self.num_tokens * self.d_model])
        output = self.output_layer(sal_en_aplanado)
        return F.softmax(output, 1)