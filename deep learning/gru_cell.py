import tensorflow as tf


class MyGRUCell(tf.keras.layers.Layer):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Reset gate
        self.W_r = self.add_weight(shape=(input_dim, self.hidden_size), initializer="glorot_uniform", trainable=True)
        self.U_r = self.add_weight(shape=(self.hidden_size, self.hidden_size), initializer="glorot_uniform", trainable=True)
        self.b_r = self.add_weight(shape=(self.hidden_size,), initializer="zeros", trainable=True)

        # Update gate
        self.W_z = self.add_weight(shape=(input_dim, self.hidden_size), initializer="glorot_uniform", trainable=True)
        self.U_z = self.add_weight(shape=(self.hidden_size, self.hidden_size), initializer="glorot_uniform", trainable=True)
        self.b_z = self.add_weight(shape=(self.hidden_size,), initializer="zeros", trainable=True)

        # Candidate hidden state
        self.W_h = self.add_weight(shape=(input_dim, self.hidden_size), initializer="glorot_uniform", trainable=True)
        self.U_h = self.add_weight(shape=(self.hidden_size, self.hidden_size), initializer="glorot_uniform", trainable=True)
        self.b_h = self.add_weight(shape=(self.hidden_size,), initializer="zeros", trainable=True)

    def call(self, inputs, hidden_state):
        r_t = tf.sigmoid(tf.matmul(inputs, self.W_r) + tf.matmul(hidden_state, self.U_r) + self.b_r)
        z_t = tf.sigmoid(tf.matmul(inputs, self.W_z) + tf.matmul(hidden_state, self.U_z) + self.b_z)
        h_tilde = tf.tanh(tf.matmul(inputs, self.W_h) + tf.matmul(r_t * hidden_state, self.U_h) + self.b_h)
        h_next = (1 - z_t) * hidden_state + z_t * h_tilde

        return h_next

# Example Usage
batch_size = 4
input_dim = 3
hidden_size = 5

gru_cell = MyGRUCell(hidden_size)

x_t = tf.random.normal((batch_size, input_dim))
h_t_prev = tf.zeros((batch_size, hidden_size))

h_t = gru_cell(x_t, h_t_prev)
print("Next GRU Hidden State:\n", h_t.numpy())
