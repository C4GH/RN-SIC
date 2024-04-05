import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader

# Carga de embeddings y vocabulario
def load_embeddings_and_vocab(embeddings_path, vocab_path):
    embeddings = np.load(embeddings_path)
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float)
    return embeddings_tensor, vocab

# Clase CustomDataset
class CustomDataset(Dataset):
    def __init__(self, json_path, embeddings_tesor, vocab):

# Función para crear el DataLoader
def get_dataloader(json_path, embeddings_tensor, vocab, batch_size=32):
    dataset = CustomDataset(json_path, embeddings_tensor, vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
