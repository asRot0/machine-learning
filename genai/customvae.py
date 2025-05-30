import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class CustomVAE(keras.Model):
    def __init__(self, input_shape, latent_dim=2, mode="mnist", **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.mode = mode.lower()
        self.encoder = self.build_encoder(input_shape)
        self.decoder = self.build_decoder(input_shape)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def get_layers_by_mode(self, x, filters):
        """Add layers based on dataset complexity."""
        if self.mode == "mnist":
            x = layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu")(x)
        elif self.mode == "fashion":
            x = layers.Conv2D(filters, 3, strides=2, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        elif self.mode == "faces":
            x = layers.Conv2D(filters, 3, strides=2, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
        return x

    def build_encoder(self, input_shape):
        inputs = keras.Input(shape=input_shape)
        x = inputs

        filters = 32
        for i in range(3 if self.mode == "faces" else 2):
            x = self.get_layers_by_mode(x, filters)
            filters *= 2

        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    def build_decoder(self, input_shape):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        size = input_shape[0] // (4 if self.mode == "mnist" else 8)
        filters = 64 if self.mode == "faces" else 32

        x = layers.Dense(size * size * filters, activation="relu")(latent_inputs)
        x = layers.Reshape((size, size, filters))(x)

        for i in range(2 if self.mode == "mnist" else 3):
            if self.mode == "faces":
                x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
                x = layers.BatchNormalization()(x)
                x = layers.LeakyReLU()(x)
            else:
                x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same", activation="relu")(x)
            filters //= 2

        outputs = layers.Conv2D(input_shape[-1], 3, activation="sigmoid", padding="same")(x)
        return keras.Model(latent_inputs, outputs, name="decoder")

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2, 3))
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
