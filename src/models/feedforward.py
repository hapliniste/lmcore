import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor=4, kernel_size=3, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.expansion_factor = expansion_factor
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv1d(dim, dim * expansion_factor, kernel_size, padding=kernel_size // 2, groups=dim)
        self.conv2 = nn.Conv1d(dim * expansion_factor, dim, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, dim)
        return x