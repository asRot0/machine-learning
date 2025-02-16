"""
Softmax Function Implementation

Formula:
    \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}

Variables:
    z_i: Raw model output (logits)
    e^{z_i}: Exponential transformation ensuring positive values
    \sum e^{z_j}: Normalization factor ensuring probabilities sum to 1
"""

import numpy as np

def softmax(z):
    """Computes the softmax probabilities for a given input array."""
    exp_z = np.exp(z - np.max(z))  # Subtract max for numerical stability
    return exp_z / np.sum(exp_z)

# Example Usage
if __name__ == "__main__":
    logits = np.array([2.0, 1.0, 0.1])
    probabilities = softmax(logits)
    print("Softmax Probabilities:", probabilities)
