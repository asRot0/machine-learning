import tensorflow as tf


class MyLSTMCell(tf.keras.layers.Layer):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Create trainable weights for all gates
        self.W_f = self.add_weight(shape=(input_dim, self.hidden_size), initializer="glorot_uniform", trainable=True)
        self.U_f = self.add_weight(shape=(self.hidden_size, self.hidden_size), initializer="glorot_uniform",
                                   trainable=True)
        self.b_f = self.add_weight(shape=(self.hidden_size,), initializer="zeros", trainable=True)

        self.W_i = self.add_weight(shape=(input_dim, self.hidden_size), initializer="glorot_uniform", trainable=True)
        self.U_i = self.add_weight(shape=(self.hidden_size, self.hidden_size), initializer="glorot_uniform",
                                   trainable=True)
        self.b_i = self.add_weight(shape=(self.hidden_size,), initializer="zeros", trainable=True)

        self.W_c = self.add_weight(shape=(input_dim, self.hidden_size), initializer="glorot_uniform", trainable=True)
        self.U_c = self.add_weight(shape=(self.hidden_size, self.hidden_size), initializer="glorot_uniform",
                                   trainable=True)
        self.b_c = self.add_weight(shape=(self.hidden_size,), initializer="zeros", trainable=True)

        self.W_o = self.add_weight(shape=(input_dim, self.hidden_size), initializer="glorot_uniform", trainable=True)
        self.U_o = self.add_weight(shape=(self.hidden_size, self.hidden_size), initializer="glorot_uniform",
                                   trainable=True)
        self.b_o = self.add_weight(shape=(self.hidden_size,), initializer="zeros", trainable=True)

    def call(self, inputs, states):
        """
        Forward pass for a single LSTM time step.

        Args:
            inputs: (batch_size, input_dim)
            states: Tuple containing (previous_hidden_state, previous_cell_state)

        Returns:
            new_hidden_state, new_cell_state
        """
        h_prev, c_prev = states

        # Forget gate
        f_t = tf.sigmoid(tf.matmul(inputs, self.W_f) + tf.matmul(h_prev, self.U_f) + self.b_f)

        # Input gate
        i_t = tf.sigmoid(tf.matmul(inputs, self.W_i) + tf.matmul(h_prev, self.U_i) + self.b_i)
        c_tilde = tf.tanh(tf.matmul(inputs, self.W_c) + tf.matmul(h_prev, self.U_c) + self.b_c)

        # Cell state update
        c_next = f_t * c_prev + i_t * c_tilde

        # Output gate
        o_t = tf.sigmoid(tf.matmul(inputs, self.W_o) + tf.matmul(h_prev, self.U_o) + self.b_o)
        h_next = o_t * tf.tanh(c_next)

        return h_next, c_next


# Example Usage
batch_size = 4
input_dim = 3
hidden_size = 5

lstm_cell = MyLSTMCell(hidden_size)

# Example input (batch_size, input_dim)
x_t = tf.random.normal((batch_size, input_dim))
h_t_prev = tf.zeros((batch_size, hidden_size))  # Initial hidden state
c_t_prev = tf.zeros((batch_size, hidden_size))  # Initial cell state

h_t, c_t = lstm_cell(x_t, (h_t_prev, c_t_prev))
print("Next Hidden State:\n", h_t.numpy())
print("Next Cell State:\n", c_t.numpy())
