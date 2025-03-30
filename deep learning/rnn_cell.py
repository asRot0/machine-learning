import tensorflow as tf


class MyRNNCell(tf.keras.layers.Layer):
    def __init__(self, hidden_size, output_size, **kwargs):
        super(MyRNNCell, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_size = output_size

    def build(self, input_shape):
        input_dim = input_shape[-1]  # Get input feature size

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

    def call(self, inputs, states):
        """
        Forward pass for a single time step.

        Args:
            inputs: Tensor of shape (batch_size, input_dim)
            states: List containing the previous hidden state (batch_size, hidden_size)

        Returns:
            output: Output tensor of shape (batch_size, output_size)
            new_states: Updated hidden state tensor (batch_size, hidden_size)
        """
        hidden_state = states[0]  # Extract previous hidden state
        h_next = tf.tanh(tf.matmul(inputs, self.W_x) + tf.matmul(hidden_state, self.W_h) + self.b_h)
        y = tf.matmul(h_next, self.W_y) + self.b_y
        return y, [h_next]  # RNN expects (output, new_state)

    @property
    def state_size(self):
        return self.hidden_size  # Required by tf.keras.layers.RNN


# Example Usage
batch_size = 4
timesteps = 6
input_dim = 3
hidden_size = 5
output_size = 2

# Initialize RNN Cell
rnn_cell = MyRNNCell(hidden_size, output_size)
rnn_layer = tf.keras.layers.RNN(rnn_cell, return_sequences=True, return_state=True)

# Generate random input sequence (batch_size, timesteps, input_dim)
X = tf.random.normal((batch_size, timesteps, input_dim))

# Forward pass using tf.keras.layers.RNN
outputs, final_state = rnn_layer(X)

print("Output Shape:", outputs.shape)  # (batch_size, timesteps, output_size)
print("Final Hidden State Shape:", final_state.shape)  # (batch_size, hidden_size)
