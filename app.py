from torch.utils.data import Dataset


class CharDataset(Dataset):
    """
    Emits batches of characters.
    Adapted from "https://github.com/karpathy/minGPT".
    """

    def __init__(self, config, data):
        chars = None  # TODO
        self.stoi = {
            ch: i for i, ch in enumerate(chars)
        }  # map characters to integer indices

    def get_vocab_size(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        # encode every character to an integer
        # return the chunk and the shifted version as tensors
        pass


# ---------------------

from module import TransformerDecoder

# Exemple d'utilisation
decoder = (TransformerDecoder(
            num_layers=6,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            output_size=1000))


f = open("example.txt")

# Exemple d'entrée
input_data = torch.randn(
            (32,    # batch_size
             20,    # seq_length
             512))  # embedding_size

# Appliquer le décodeur
output = decoder(input_data)
print("Output shape:", output.shape)
