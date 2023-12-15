from module import TransformerDecoder
import pandas as pd
import torch

# Exemple d'utilisation
decoder = (TransformerDecoder(
            num_layers=6,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            output_size=1000))


# Exemple d'entrée
input_data = torch.randn(
            (100,    # batch_size
             10,    # seq_length
             512))  # embedding_size

# Appliquer le décodeur
output = decoder(input_data)
print("Output shape:", output.shape)
