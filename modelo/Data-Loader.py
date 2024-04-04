import torch
from torch.utils.data import Dataset, DataLoader
import json

class CustomDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_tensor = self.preprocess(item['input'])
        target_tensor = self.preprocess(item['input'])
        return input_tensor, target_tensor

    def preprocess(self, text):
        return torch.tensor([0])
