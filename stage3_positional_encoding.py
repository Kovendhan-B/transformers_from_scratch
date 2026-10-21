
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(max_len).unsqueeze(1).float()

        # even dimensions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        # Even: sine
        pe[:, 0::2] = torch.sin(position * div_term)

        # Odd: cosine
        pe[:, 1::2] = torch.cos(position * div_term)

        
        pe = pe.unsqueeze(0)

        
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    d_model = 8

    x = torch.zeros(batch_size, seq_len, d_model)

    positional_encoding = PositionalEncoding(
        d_model=d_model,
        max_len=100
    )

    result = positional_encoding(x)

    print("Input shape:", x.shape)
    print("Positional encoding shape:", positional_encoding.pe.shape)
    print("Output shape:", result.shape)
    print("Position 0:", result[0, 0])
    print("Position 1:", result[0, 1])

    assert result.shape == (2, 5, 8)
    assert not torch.equal(result[:, 0, :], result[:, 1, :])

    print("Stage 3 passed!")
