"""
Gradient Descent Implementation

Formula:
    \theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)

Variables:
    \theta_t: Model parameters at step t
    \eta: Learning rate (step size)
    \nabla J(\theta_t): Gradient of the cost function
"""

import numpy as np

def gradient_descent(X, y, theta, learning_rate=0.01, epochs=1000):
    """Performs gradient descent optimization."""
    m = len(y)
    for _ in range(epochs):
        gradient = (1/m) * X.T @ (X @ theta - y)
        theta -= learning_rate * gradient
    return theta

# Example Usage
if __name__ == "__main__":
    X = np.array([[1, 1], [1, 2], [1, 3]])  # Adding bias term
    y = np.array([2, 2.5, 3.5])
    theta = np.zeros(X.shape[1])
    theta = gradient_descent(X, y, theta)
    print("Optimized Parameters:", theta)
