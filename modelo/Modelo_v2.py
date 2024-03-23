import torch
import torch.nn as nn
from torch import Tensor

from embedding import TokenEmbeddingSRC as tesrc
from modelo import PositionalEncoding as pose

src_vocab_size = 1627916
emb_size = 100
ArchivoEMB = 'C:\\Users\\afloresre\\Documents\\Cagh\\Red\\salida\\embs_npa800.npy'

# Cargar los embeddings
src_tok_emb = tesrc(src_vocab_size, emb_size, ArchivoEMB)


class Modelo(nn.Module):
    def __init__(self, nhead: int, num_layers: int, emb_size: int, dropout: float):
        super(Modelo, self).__init__()
        #self.src_tok_emb = src_tok_emb
        self.positional_encoding = pose(emb_size, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src_emb: Tensor):
            #src_emb = self.src_tok_emb.emb(src)
            src_emb_pe=self.positional_encoding(src_emb)
            output=self.transformer_encoder(src_emb_pe)
            output_flattened = output.view(-1)
            return output_flattened

nhead = 3
num_layers = 4
dropout = 0.05

modelo = Modelo( nhead=nhead, num_layers=num_layers, emb_size=emb_size, dropout=dropout)

src=[2345,4456,34,.......] #vector de tokens
src_emb = src_tok_emb.emb(src) # se calcula el embedding aqui afuera 

modelo.forward(src_emb)
