import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float().to(device)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / d_model)
        ).to(device)
        self.positional_encoding = nn.Parameter(
                torch.zeros(1, max_len, d_model))
        with torch.no_grad():
            self.positional_encoding[:, :, 0::2] = torch.sin(position * div_term)
            self.positional_encoding[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        # x : (batch_size, sequence_length, d_model)
        res = self.positional_encoding[:, : x.size(1)].detach()
        return x + res


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(CausalSelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model).to(device)
        self.v_linear = nn.Linear(d_model, d_model).to(device)
        self.k_linear = nn.Linear(d_model, d_model).to(device)

        self.out_linear = nn.Linear(d_model, d_model).to(device)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.q_linear(x).view(batch_size, seq_len,
                                  self.n_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_len,
                                  self.n_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_len,
                                  self.n_heads, self.head_dim)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, self.head_dim)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, self.head_dim)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, self.head_dim)

        attn_scores = torch.matmul(q, k.transpose(1, 2)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )
        attn_scores = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_scores, v)

        out = (
            out.view(self.n_heads, batch_size, seq_len, self.head_dim)
            .permute(1, 2, 0, 3)
            .contiguous()
        )
        out = out.view(batch_size, seq_len, -1)

        out = self.out_linear(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.linear2 = nn.Linear(d_ff, d_model).to(device)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.CausalSelfAttn = CausalSelfAttention(d_model, n_heads).to(device)
        self.LayerNorm_1 = nn.LayerNorm(d_model).to(device)
        self.MLP = FeedForward(d_model, d_ff, dropout).to(device)
        self.LayerNorm_2 = nn.LayerNorm(d_model).to(device)

    def forward(self, x):
        x = x + self.CausalSelfAttn(self.LayerNorm_1(x))
        out = x + self.MLP(self.LayerNorm_2(x))
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers,
                 n_heads, d_ff, max_len=512, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.WTE = nn.Embedding(vocab_size, d_model).to(device)
        self.WPE = PositionalEncoding(d_model, max_len).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

        self.Blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout)
             for _ in range(n_layers)]
        )

        self.Final_LayerNorm = nn.LayerNorm(d_model).to(device)
        self.LM_Head = nn.Linear(d_model, vocab_size).to(device)

    def forward(self, idx):
        # pos = torch.arange(0, T).unsqueeze(0).repeat(B, 1)
        tok_emb = self.WTE(idx)
        pos_emb = self.WPE(tok_emb)
        x = self.dropout(tok_emb + pos_emb)
        for block in self.Blocks:
            x = block(x)
        x = self.Final_LayerNorm(x)
        res = self.LM_Head(x)  # logit
        return res

    def generate(self, seed_tokens, num_tokens_to_generate):
        self.eval()

        with torch.no_grad():
            tok_emb = self.WTE(seed_tokens)
            pos_emb = self.WPE(tok_emb)
            x = self.dropout(tok_emb + pos_emb)

            for block in self.Blocks:
                x = block(x)

            x = self.Final_LayerNorm(x)
            generated_tokens = self.LM_Head(x)
            generated_tokens = generated_tokens[:, -num_tokens_to_generate:]
        self.train()
        return generated_tokens
