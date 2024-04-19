import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json
from pympler import asizeof
import math

# Definición del CustomDataset
class CustomDataset(Dataset):
    def __init__(self, json_path, embeddings_path, vocab_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.embeddings = torch.tensor(np.load(embeddings_path), dtype=torch.float)
        vocab_words = np.load(vocab_path)
        self.vocab = {word: idx for idx, word in enumerate(vocab_words)}
        if '<unk>' not in self.vocab:
            raise ValueError("El token '<unk>' debe de estar en el vocabulario.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item[0]
        indices = [self.vocab.get(word, self.vocab.get('<unk>')) for word in text.split()]
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        label = torch.tensor(item[1], dtype=torch.float)
        return indices_tensor, label

def collate_fn(batch):
    indices_list, labels_list = zip(*batch)
    padded_indices = pad_sequence(indices_list, batch_first=True, padding_value=0)

    max_length = padded_indices.shape[1]

    # Pad labels to the maximum label length in the batch
    max_label_length = max(label.size(0) for label in labels_list)
    padded_labels = torch.zeros((len(labels_list), max_label_length), dtype=torch.float)
    for i, label in enumerate(labels_list):
        padded_labels[i, :label.size(0)] = label

    return padded_indices, padded_labels

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
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.output_layer = nn.Linear(embedding_dim * num_tokens, num_classes)

    def forward(self, emb1):
        emb1 = self.embedding(emb1) # Convierte indices a embeddings
        print(f"Después de embedding: {emb1.shape}")
        emb1 = self.positional_encoding(emb1)
        emb1 = self.transformer_encoder(emb1)
        print(f"Después de transformer encoder: {emb1.shape}")
        emb1 = emb1.view(emb1.size(0), -1) # Aplana salida para capa lineal
        print(f"Después de aplanar: {emb1.shape}")
        # Revisa si el aplanado coincide con el caso de entrada de la capa lineal
        expected_size = self.d_model * self.num_tokens
        if emb1.size != expected_size:
            raise ValueError(f'Expected size {expected_size}, but got {emb1.size(1)}')
        emb1 = self.output_layer(emb1)
        return F.softmax(emb1, dim=1)

# Función principal para ejecutar el modelo
if __name__ == '__main__':
    # Rutas a los archivos necesarios
    embeddings_path = r"C:\Users\afloresre\Documents\Cagh\Red\salida\embeddings 300\embs_npa.npy"
    vocab_path = r"C:\Users\afloresre\Documents\Cagh\Red\salida\embeddings 300\vocab_npa.npy"
    json_path = r'C:\Users\afloresre\Documents\Cagh\Red\salida\embeddings 300\salida_prueba.json'

    # Inicialización del dataset
    dataset = CustomDataset(json_path, embeddings_path, vocab_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Inicialización del modelo
    vocab_size = 1866360
    embedding_dim = 300
    d_model = embedding_dim
    model = Modelo(vocab_size, embedding_dim, d_model, 4, 3, 200, 3, 0.05)
    # Bucle para entrenar el modelo (aunque falta el optimizador y detalles de entrenamiento)
    for indices, labels in dataloader:
        # Aquí debes agregar el código para entrenar tu modelo
        predictions = model(indices)
        print(predictions.size())
