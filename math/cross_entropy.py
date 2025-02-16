"""
Cross-Entropy Loss Implementation

Formula:
    \mathcal{L} = -\sum_{i=1}^{n} y_i \log(\hat{y_i})

Variables:
    y_i: True class label (ground truth)
    \hat{y_i}: Predicted probability for class i
"""

import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """Computes the cross-entropy loss given true labels and predicted probabilities."""
    epsilon = 1e-12  # To prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  # Clipping for numerical stability
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)

# Example Usage
if __name__ == "__main__":
    y_true = np.array([1, 0, 0])  # One-hot encoded
    y_pred = np.array([0.7, 0.2, 0.1])  # Predicted probabilities
    loss = cross_entropy_loss(y_true, y_pred)
    print("Cross-Entropy Loss:", loss)
