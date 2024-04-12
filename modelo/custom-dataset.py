import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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

        if '<unk>' not in self.vocab:
            raise ValueError("El token '<unk>' debe de estar en el vocabulario.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # El texto a procesar está en el primer elemento de 'item'
        text = item[0] # asumiendolo cadena de texto
        indices = [self.vocab.get(word, self.vocab.get('<unk>')) for word in text.split()]
        indices_tensor = torch.tensor(indices, dtype=torch.long)
       # Usando las probabilidades como etiqueta
        label = torch.tensor(item[1], dtype=torch.float)

        return indices_tensor, label

def collate_fn(batch):
    # descomponer el batch en listas de índices y etiquetas
    indices_list, labels_list = zip(*batch)

    # agrega paddinbg a la lista de índices para que todas tengan la misma longitud
    padded_indices = pad_sequence(indices_list,batch_first=True, padding_value=0)

    # Apilar etiquetas en un tensor
    labels = torch.stack(labels_list)
    return padded_indices, labels

if __name__ == '__main__':
    embeddings_path = r"C:\Users\afloresre\Documents\Cagh\Red\salida\embs_npa800.npy"
    vocab_path = r"C:\Users\afloresre\Documents\Cagh\Red\salida\vocab_npa800.npy"
    json_path = r"C:\Users\afloresre\Documents\Cagh\Red\salida.json"

    # Inicialización del dataset
    dataset = CustomDataset(json_path, embeddings_path, vocab_path)

    # Medición y visualización del tamaño total en memoria del dataset
    total_size = asizeof.asizeof(dataset)
    print(f"Tamaño del dataset en memoria: {total_size} bytes")

    # Creación del DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Iteración sobre el dataloader
    for embeddings, labels in dataloader:
        print(embeddings.size(), labels.size())

    # Luego pongo la lógica de entrenamiento y evaluación.

    