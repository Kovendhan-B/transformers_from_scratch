
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=100):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model, max_len
        )

    def forward(self, token_ids):
        # token_ids: (batch_size, seq_len)
        embeddings = self.token_embedding(token_ids)

        # Multiply embeddings by sqrt(d_model)
        embeddings = embeddings * math.sqrt(
            self.token_embedding.embedding_dim
        )

        return self.positional_encoding(embeddings)


if __name__ == "__main__":
    torch.manual_seed(42)

    vocab_size = 1000
    d_model = 8

    
    token_ids = torch.tensor([
        [12, 45, 67, 89, 23],
        [31, 54, 76, 18, 92]
    ])

    model = InputEmbedding(vocab_size, d_model)
    output = model(token_ids)

    print("Token IDs:", token_ids.shape)
    print("Output embeddings:", output.shape)

    assert output.shape == (2, 5, 8)
    assert torch.isfinite(output).all()

    print("Stage 4passed!")
