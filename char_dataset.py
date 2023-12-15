import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, data_path, block_size):
        with open(data_path, "r", encoding="utf-8") as file:
            text = file.read()

        # Extract unique characters from the text
        chars = list(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size

        # Encode the entire text into integers
        self.data = [self.stoi[ch] for ch in text]

    def get_vocab_size(self):
        return len(self.stoi)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Get a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx: idx + self.block_size + 1]

        # Convert the chunk to a PyTorch tensor
        chunk_tensor = torch.tensor(chunk, dtype=torch.long)

        # Return the chunk and the shifted version as tensors
        return chunk_tensor[:-1], chunk_tensor[1:]


# Exemple d'utilisation avec les œuvres de Shakespeare
# shakespeare_dataset = CharDataset("shakespeare.txt", block_size=128)
# vocab_size = shakespeare_dataset.get_vocab_size()
# print("Vocabulaire Size:", vocab_size)
# print("Longueur de l'ensemble de données:", len(shakespeare_dataset))
# 
# # Obtenez un exemple du dataset
# sample_input, sample_target = shakespeare_dataset[0]
# print("Exemple d'entrée:", sample_input.shape)
# print("Exemple de cible:", sample_target.shape)
