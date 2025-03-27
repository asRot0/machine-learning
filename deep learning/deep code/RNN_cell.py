import numpy as np


class MyRNNCell:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the RNN cell with weight matrices and biases.
        """
        self.hidden_size = hidden_size

        # Xavier/Glorot initialization
        self.W_x = np.random.randn(hidden_size, input_size) * np.sqrt(1 / input_size)
        self.W_h = np.random.randn(hidden_size, hidden_size) * np.sqrt(1 / hidden_size)
        self.b_h = np.zeros((hidden_size, 1))

        self.W_y = np.random.randn(output_size, hidden_size) * np.sqrt(1 / hidden_size)
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x, h_prev):
        """
        Forward pass through a single time step of the RNN.

        Args:
            x (numpy array): Input vector of shape (input_size, 1)
            h_prev (numpy array): Previous hidden state of shape (hidden_size, 1)

        Returns:
            h_next: Updated hidden state
            y: Output of the current step
        """
        h_next = np.tanh(np.dot(self.W_x, x) + np.dot(self.W_h, h_prev) + self.b_h)
        y = np.dot(self.W_y, h_next) + self.b_y
        return h_next, y


# Example usage
input_size = 3
hidden_size = 5
output_size = 2
rnn_cell = MyRNNCell(input_size, hidden_size, output_size)

# Example input (batch size = 1)
x_t = np.random.randn(input_size, 1)
h_t_prev = np.zeros((hidden_size, 1))  # Initial hidden state

h_t, y_t = rnn_cell.forward(x_t, h_t_prev)
print("Next Hidden State:\n", h_t)
print("Output:\n", y_t)
