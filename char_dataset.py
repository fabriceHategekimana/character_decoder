import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, data_path, block_size):
        with open(data_path, "r", encoding="utf-8") as file:
            text = file.read()

        chars = list(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.data = [self.stoi[ch] for ch in text]

    def get_vocab_size(self):
        return len(self.stoi)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        chunk_tensor = torch.tensor(chunk, dtype=torch.long)
        return chunk_tensor[:-1], chunk_tensor[1:]

    def to_string(self, integers):
        return "".join([self.itos[e.item()] for e in integers])

    def to_integer(self, string):
        return torch.tensor([self.stoi[e] for e in string])

