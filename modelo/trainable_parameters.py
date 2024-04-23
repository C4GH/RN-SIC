import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Modelo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, d_model, nhead, num_layers, num_tokens, num_classes, dropout=0.05,
                 freeze_embedding=False):
        super(Modelo, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Control the trainability of the embedding layer
        for param in self.embedding.parameters():
            param.requires_grad = not freeze_embedding  # Set True if not freezing, False if freezing

        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model * num_tokens, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.view(x.size(0), -1)
        return self.output_layer(x)


if __name__ == '__main__':
    # Model parameters
    vocab_size = 1866360
    embedding_dim = 300
    d_model = 300
    num_tokens = 200
    num_classes = 3

    # Initialize model, ensuring embedding is trainable
    model = Modelo(vocab_size, embedding_dim, d_model, 4, 3, num_tokens, num_classes, 0.05, freeze_embedding=False)

    # Print out the trainability of each parameter
    print("Model parameters and their trainability status:")
    for name, param in model.named_parameters():
        print(f"{name}: {'trainable' if param.requires_grad else 'frozen'}")
