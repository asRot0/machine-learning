"""
Transformer Attention Mechanism Implementation

Formula:
    Attention(Q, K, V) = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right) V

Variables:
    Q: Query matrix
    K: Key matrix
    V: Value matrix
    d_k: Dimensionality of key vectors (scaling factor)
    QK^T: Dot product of queries and keys to compute attention scores
    softmax: Normalization to ensure values sum to 1
"""

import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """Computes the scaled dot-product attention."""
    d_k = Q.shape[-1]  # Dimensionality of key vectors
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)  # Compute scaled scores
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)  # Apply softmax
    output = np.matmul(attention_weights, V)  # Compute weighted sum of values
    return output, attention_weights

# Example Usage
if __name__ == "__main__":
    Q = np.array([[1, 0, 1], [0, 1, 0]])  # Example query matrix
    K = np.array([[1, 2, 1], [1, 1, 0], [0, 1, 1]])  # Example key matrix
    V = np.array([[1, 0], [0, 1], [1, 1]])  # Example value matrix
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    print("Attention Output:\n", output)
    print("Attention Weights:\n", attention_weights)
