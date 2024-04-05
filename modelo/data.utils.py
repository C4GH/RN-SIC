import json
import numpy as np
import torch
from torch.utils.data import Dataset

def load_embeddings(embeddings_path):
    embeddings = np.load(embeddings_path)
    return torch.tensor(embeddings, dtype=torch.float)

def load_vocab(vocab_path):
    vocab_words = np.load(vocab_path)
    return {word: idx for idx, word in enumerate(vocab_words)}

class CustomDataset(Dataset):
    def __init__(self, json_path, embeddings_tensor, vocab):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.embeddings = embeddings_tensor
        self.vocab = vocab

    def word_to_index(self, word):
        return self.vocab.get(word, self.vocab.get('UNK', 0))

    def __getitem__(self, idx):
        item = self.data[idx]
        indices = [self.word_to_index(word) for word in item['texto'].split()]
        embedding_tensor = self.embeddings[torch.tensor(indices, dtype=torch.long)]
        return embedding_tensor, torch.tensor(item['etiqueta'])

    def __len__(self):
        return len(self.data)