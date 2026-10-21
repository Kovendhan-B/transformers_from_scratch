
import torch
import torch.nn as nn


class PositionWiseFeedFwd(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    d_model = 8
    d_ff = 32

    x = torch.randn(batch_size, seq_len, d_model)

    ffn = PositionWiseFeedFwd(d_model, d_ff)
    output = ffn(x)

    print("Input shape:", x.shape)
    print("Hidden layer shape:", ffn.linear1(x).shape)
    print("Output shape:", output.shape)

    assert output.shape == (2, 5, 8)
    assert torch.isfinite(output).all()

    print("Stage 5 passed!")
