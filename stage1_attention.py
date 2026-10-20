import math
import torch

def scaled_dot_product_attention(Q,K,V):
    """
    Q: (seq_len_q,dk)
    K: (seq_len_k,dk)
    V: (seq_len_k,dk)
    """
    
    # Query Key similarity
    scores= Q @ K.transpose(-2,-1)
    
    # Scale scores by root of key dim
    dk=Q.size(-1)
    scores=scores/math.sqrt(dk)
    
    # Normalise scores
    weights = torch.softmax(scores, dim=-1)
    
    #OP
    output = weights @ V 
    return output,weights

if __name__ == "__main__":
    torch.manual_seed(42)

    Q = torch.randn(3, 4)
    K = torch.randn(3, 4)
    V = torch.randn(3, 4)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("Q shape:", Q.shape)
    print("K shape:", K.shape)
    print("V shape:", V.shape)

    print("\nAttention weights:")
    print(weights)

    print("\nOutput:")
    print(output)

    print("\nOutput shape:", output.shape)
    print("Row sums:", weights.sum(dim=-1))