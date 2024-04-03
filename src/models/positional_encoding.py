import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))

    def forward(self, x, offset=0):
        batch_size, seq_len = x.shape[0], x.shape[1]
        position = torch.arange(offset, offset + seq_len, dtype=torch.long, device=x.device)
        sin_enc = torch.einsum("i,j->ij", position, self.inv_freq)
        cos_enc = torch.cos(sin_enc)
        sin_enc = torch.sin(sin_enc)
        enc = torch.stack([sin_enc, cos_enc], dim=-1).flatten(1)
        enc = enc.repeat(batch_size, 1, 1)
        return enc