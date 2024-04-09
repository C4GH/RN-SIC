import  torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pympler import asizeof

class CustomDataset(Dataset):
    def __init__(self, json_path, embeddings_path, vocab_path):
        # Carga del JSON que contiene los datos
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # Carga de embeddings y vocabulario
        self.embeddings = torch.tensor(np.load(embeddings_path), dtype=torch.float)
        vocab_words = np.load(vocab_path)
        self.vocab = {word: idx for idx, word in enumerate(vocab_words)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # convertir el texto a índices usando el vocabulario
        indices = [self.vocab.get(word, self.vocab.get('UNK', 0)) for word in item['texto'].split()]

        # Obtener los embeddings correspondientes a los índices
        embeddings = self.embeddings[torch.tensor(indices, dtype=torch.long)]

        label = torch.tensor(item['etiqueta'], dtype=torch.long)

        return embeddings, label

if __name__ == '__main__':
    embeddings_path = r"C:\Users\afloresre\Documents\Cagh\Red\salida\embs_npa800.npy"
    vocab_path = r"C:\Users\afloresre\Documents\Cagh\Red\salida\vocab_npa800.npy"
    json_path = r"C:\Users\afloresre\Documents\Cagh\Red\salida.json"

    # Creación del dataset
    dataset = CustomDataset(json_path, embeddings_path, vocab_path)
    total_size = asizeof.asizeof(dataset)
    print(f"Tamaño del dataset en memoria: {total_size} bytes")

    # Creación del DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    #for embeddings, labels in dataloader:
     #   print(embeddings.size(), labels.size())