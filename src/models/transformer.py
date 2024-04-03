import torch.nn as nn
from models.positional_encoding import RotaryPositionalEmbedding
from models.attention import MultiHeadAttention
from models.feedforward import FeedForward

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, dim, max_len, num_heads, num_layers, dropout):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, dim)
        self.positional_encoding = RotaryPositionalEmbedding(dim, max_len)
        self.layers = nn.ModuleList([
            TransformerLayer(dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        token_embeddings = self.token_embeddings(input_ids)
        positional_embeddings = self.positional_encoding(token_embeddings)
        x = token_embeddings + positional_embeddings
        for layer in self.layers:
            x = layer(x)
        return x
    
    def generate(self, input_ids, max_length, num_beams, early_stopping, top_k, top_p, do_sample, num_return_sequences):
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences
        )
        return outputs

class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads)
        self.ffn = FeedForward(dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x