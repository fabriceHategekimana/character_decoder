import torch
import torch.nn as nn

# Use the Adam optimizer and the cross-entropy loss.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # d_model = 512
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # query.shape: torch.Size([32, 20, 300])
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (640x300 and 512x512)
        batch_size, _, _ = query.shape
        Q = (
            # linear_q = (in:512, out:512)
            self.linear_q(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.linear_k(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.linear_v(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        out = (
            torch.matmul(attention_weights, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim)
        )
        out = self.linear_out(out)
        return out


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # d_model = 512
        # num_heads = 8
        # d_ff = 2048
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)  # reduce overfitting

        self.feedforward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # File ~/sessions/projet/deep_learning/projet/module.py:84, in DecoderLayer.forward(self, x, mask)
        # x.shape: torch.Size([32, 20, 300])
        attention_output = self.self_attention(x, x, x, mask)
        x = x + self.dropout1(attention_output)
        x = self.norm1(x)

        feedforward_output = self.feedforward(x)
        x = x + self.dropout2(feedforward_output)
        x = self.norm2(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff,
                 output_size, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        # d_model: 512
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout)
             for _ in range(num_layers)]
        )
        self.output_projection = nn.Linear(d_model, output_size)

    def forward(self, x, mask=None):
        # x.shape: torch.Size([32, 20, 300])
        for layer in self.layers:
            x = layer(x, mask)
        output = self.output_projection(x)
        return output
