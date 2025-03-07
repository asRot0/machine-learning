"""
Adam Optimizer Implementation

Formula:
    \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t

Variables:
    \theta_t: Model parameters at step t
    m_t: First moment estimate (mean of gradients)
    v_t: Second moment estimate (variance of gradients)
    \eta: Learning rate
    \epsilon: Small constant to avoid division by zero
"""

import numpy as np


def adam_optimizer(gradients, theta, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Performs one step of the Adam optimization algorithm."""
    m = beta1 * m + (1 - beta1) * gradients  # Update biased first moment estimate
    v = beta2 * v + (1 - beta2) * (gradients ** 2)  # Update biased second moment estimate

    m_hat = m / (1 - beta1 ** t)  # Bias-corrected first moment estimate
    v_hat = v / (1 - beta2 ** t)  # Bias-corrected second moment estimate

    theta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)  # Update parameters

    return theta, m, v


# Example Usage
if __name__ == "__main__":
    theta = np.array([0.5, 0.5])  # Initial parameters
    gradients = np.array([0.1, -0.2])  # Example gradients
    m, v = np.zeros_like(theta), np.zeros_like(theta)  # Initialize moment estimates
    t = 1  # Time step

    theta, m, v = adam_optimizer(gradients, theta, m, v, t)
    print("Updated Parameters:", theta)
