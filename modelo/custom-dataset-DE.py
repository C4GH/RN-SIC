import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json
from pympler import asizeof
import time

class CustomDataset(Dataset):
    def __init__(self, json_path, embeddings_path, vocab_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.embeddings = torch.tensor(np.load(embeddings_path), dtype=torch.float)
        vocab_words = np.load(vocab_path)
        self.vocab = {word: idx for idx, word in enumerate(vocab_words)}

        if '<unk>' not in self.vocab:
            raise ValueError("El token '<unk>' debe estar en el vocabulario.")

        # Calcular la longitud máxima de las secuencias en el dataset
        self.max_seq_length = max(len(item[0].split()) for item in self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item[0]
        indices = [self.vocab.get(word, self.vocab['<unk>']) for word in text.split()]
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        label = torch.tensor(item[1], dtype=torch.float)
        return indices_tensor, label

def collate_fn(batch, max_length):
    indices_list, labels_list = zip(*batch)
    padded_indices = pad_sequence(indices_list, batch_first=True, padding_value=0)
    # Rellenar hasta la longitud máxima del dataset
    if padded_indices.shape[1] < max_length:
        padding_size = max_length - padded_indices.shape[1]
        padded_indices = torch.cat([padded_indices, torch.zeros(len(padded_indices), padding_size, dtype=torch.long)], dim=1)

    # Pad labels to the maximum label length in the batch
    max_label_length = max(label.size(0) for label in labels_list)
    padded_labels = torch.zeros((len(labels_list), max_label_length), dtype=torch.float)
    for i, label in enumerate(labels_list):
        padded_labels[i, :label.size(0)] = label

    return padded_indices, padded_labels

if __name__ == '__main__':
    embeddings_path = r"C:\Users\Carlos Hurtado\PycharmProjects\Red SIC\Vocabulario\embs_npa.npy"
    vocab_path = r"C:\Users\Carlos Hurtado\PycharmProjects\Red SIC\Vocabulario\vocab_npa.npy"
    json_path = r"C:\Users\Carlos Hurtado\PycharmProjects\Red SIC\Vocabulario\salida_prueba.json"

    dataset = CustomDataset(json_path, embeddings_path, vocab_path)
    total_size_bytes = asizeof.asizeof(dataset)
    total_size_megabytes = total_size_bytes / (1024 ** 2)
    print(f"Tamaño del dataset en memoria: {total_size_megabytes:.2f} MB")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: collate_fn(x, dataset.max_seq_length))

    start_time = time.time()
    total_batches = 0
    try:
        for embeddings, labels in dataloader:
            total_batches += 1
            print(embeddings.size(), labels.size())
    except Exception as e:
        print(f"Ha ocurrido un error: {str(e)}")
    end_time = time.time()

    total_time = end_time - start_time
    average_time = total_time / total_batches if total_batches > 0 else 0

    print(f"Tiempo total de ejecución: {total_time:.2f} segundos")
    print(f"Tiempo promedio por batch: {average_time:.2f} segundos")
