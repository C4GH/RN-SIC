import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

class CustomDataset(Dataset):
    def __init__(self, json_path, embeddings_path, vocab_path):
        # Carga del JSON que tiene los datos
        with open(json_path, 'r') as f:
            self.data = json_path(f)

    # Carga de los embeddings y vocabulario
    self.embeddings = torch.tensor(np.load(embeddings_path), dtype=torch.float)
    vocab_words = np.load(vocab_path)
    self.vocab = {word: idx for idx, word in enumerate(vocab_words)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Obtener un item de los datos
        item = self.data[idx]

        # Convertir el texto a índices usando el vocabulario
        indices = [self.vocab.get(word, self.vocab.get('UNK', 0)) for word in item['texto'].split()]

        # Obtener los embeddings correspondientes a los índices
        embeddings = self.embeddings[torch.tensor(indices, dtype=torch.long)]

        # Suponiendo que 'etiqueta' es la clave para la etiqueta en tus datos
        label = torch.tensor(item['etiqueta'], dtype=torch.long)

        return embeddings, label

# Creación del Data Loader

if __name__ == 'main':
    dataset = CustomDataset('path/to/json', 'path/to/embeddings', 'path/to/vocab')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Iteración sobre el DataLoader
    for embeddings, labels, in dataloader:
        print(embeddings.size(), labels.size())

