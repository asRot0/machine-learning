import tensorflow as tf


class MyRNNCell(tf.keras.layers.Layer):
    def __init__(self, hidden_size, output_size, **kwargs):
        super(MyRNNCell, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_size = output_size

    def build(self, input_shape):
        input_dim = input_shape[-1]  # Get the input feature size

        # Initialize weight matrices
        self.W_x = self.add_weight(shape=(input_dim, self.hidden_size),
                                   initializer="glorot_uniform",
                                   trainable=True)
        self.W_h = self.add_weight(shape=(self.hidden_size, self.hidden_size),
                                   initializer="glorot_uniform",
                                   trainable=True)
        self.b_h = self.add_weight(shape=(self.hidden_size,),
                                   initializer="zeros",
                                   trainable=True)

        self.W_y = self.add_weight(shape=(self.hidden_size, self.output_size),
                                   initializer="glorot_uniform",
                                   trainable=True)
        self.b_y = self.add_weight(shape=(self.output_size,),
                                   initializer="zeros",
                                   trainable=True)

    def call(self, inputs, hidden_state):
        """
        Forward pass for a single time step.

        Args:
            inputs: Tensor of shape (batch_size, input_dim)
            hidden_state: Tensor of shape (batch_size, hidden_size)

        Returns:
            h_next: Next hidden state of shape (batch_size, hidden_size)
            y: Output of shape (batch_size, output_size)
        """
        h_next = tf.tanh(tf.matmul(inputs, self.W_x) + tf.matmul(hidden_state, self.W_h) + self.b_h)
        y = tf.matmul(h_next, self.W_y) + self.b_y
        return h_next, y


# Example Usage
batch_size = 4
input_dim = 3
hidden_size = 5
output_size = 2

# Initialize RNN Cell
rnn_cell = MyRNNCell(hidden_size, output_size)

# Test input and initial hidden state
x_t = tf.random.normal((batch_size, input_dim))
h_t_prev = tf.zeros((batch_size, hidden_size))

# Forward pass
h_t, y_t = rnn_cell(x_t, h_t_prev)
print("Next Hidden State:\n", h_t.numpy())
print("Output:\n", y_t.numpy())
