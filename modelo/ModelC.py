import torch
import torch.nn as nn
import math


# Definición de PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding.unsqueeze(-2))

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# Definición del modelo de red neuronal
class Modelo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, d_model, nhead, num_layers, num_tokens, num_classes, dropout=0.05):
        super(Modelo, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = False
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model * num_tokens, num_classes)

    def forward(self, input_indices):
        # Convert indices to embeddings
        emb1 = self.embedding(input_indices)
        # Add positional encoding
        emb1 = self.positional_encoding(emb1)
        # Apply the transformer encoder
        emb1 = self.transformer_encoder(emb1)
        # Flatten the output for the linear layer
        emb1 = emb1.view(emb1.size(0), -1)
        # Check if the flattened size matches the expected size
        expected_size = self.embedding_dim * self.num_tokens
        actual_size = emb1.size(1)
        if actual_size != expected_size:
            raise ValueError(f'Expected size {expected_size}, but got {actual_size}')
        # Calculate the output of the linear layer
        output = self.output_layer(emb1)
        return output
