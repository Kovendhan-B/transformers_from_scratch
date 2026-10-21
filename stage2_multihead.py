
import torch
import torch.nn as nn
import math


def scaled_dot_product_attention(Q, K, V):
    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(Q.size(-1))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ V
    return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads


        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Learnable final projection.
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape

        x = x.reshape(
            batch_size, seq_len, self.num_heads, self.d_k
        )

        return x.transpose(1, 2)

    def forward(self, x):
        # x: (batch, seq_len, d_model)

        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        head_output, attention_weights = scaled_dot_product_attention(
            Q, K, V
        )

        # Merge heads
        head_output = head_output.transpose(1, 2).contiguous()

        batch_size, seq_len, _, _ = head_output.shape

        concat_output = head_output.reshape(
            batch_size, seq_len, self.d_model
        )

        output = self.W_o(concat_output)

        return output, attention_weights


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 3
    d_model = 8
    num_heads = 2

    x = torch.randn(batch_size, seq_len, d_model)

    model = MultiHeadAttention(d_model, num_heads)

    output, weights = model(x)

    print("Input:", x.shape)
    print("Output:", output.shape)
    print("Attention weights:", weights.shape)

    assert output.shape == (2, 3, 8)
    assert weights.shape == (2, 2, 3, 3)

    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones(batch_size, num_heads, seq_len),
        atol=1e-6,
    )

    print("Stage 2passed!")
