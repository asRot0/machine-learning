import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Sampling(layers.Layer):
    """
    Custom Keras layer that performs the reparameterization trick.
    It takes in the latent distribution parameters (mean and log variance),
    and outputs a sampled latent vector z using:
        z = mean + exp(0.5 * log_var) * epsilon
    This enables backpropagation through stochastic sampling.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]     # dynamic batch size
        dim = tf.shape(z_mean)[1]       # latent dimension
        epsilon = tf.random.normal(shape=(batch, dim))  # random noise
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    """
    Variational Autoencoder subclassing keras.Model.
    It includes an encoder, a decoder, and custom training logic using `train_step()`.
    """
    def __init__(self, input_shape=(28, 28, 1), latent_dim=2, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim  # dimension of latent space

        # Build encoder and decoder using helper methods
        self.encoder = self.build_encoder(input_shape)
        self.decoder = self.build_decoder(input_shape)

        # Define trackers for logging different components of the loss
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def build_encoder(self, input_shape):
        """
        Builds the encoder model that maps input images to (z_mean, z_log_var, z).
        The output is a latent representation sampled using the reparameterization trick.
        """
        inputs = keras.Input(shape=input_shape, name="encoder_input")
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    def build_decoder(self, input_shape):
        """
        Builds the decoder model that reconstructs the input from latent space z.
        The output shape matches the input image shape.
        """
        inputs = keras.Input(shape=(self.latent_dim,), name="decoder_input")
        x = layers.Dense(7 * 7 * 64, activation="relu")(inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        return keras.Model(inputs, outputs, name="decoder")

    def compile(self, optimizer):
        """
        Override compile to store optimizer for custom training.
        This is necessary because we're not using compile(loss=...) like in standard models.
        """
        super(VAE, self).compile()
        self.optimizer = optimizer

    @property
    def metrics(self):
        """
        Returns a list of metrics to reset and track during each epoch.
        This enables live reporting of metrics during `model.fit(...)`.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def train_step(self, data):
        """
        Custom training loop using GradientTape.
        Calculates VAE loss = reconstruction loss + KL divergence.
        """
        if isinstance(data, tuple):
            data = data[0]  # discard labels if present

        with tf.GradientTape() as tape:
            # Forward pass through encoder and decoder
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Compute pixel-wise binary crossentropy reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2, 3)  # sum over image dimensions
                )
            )

            # Compute KL divergence (regularization loss)
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1  # sum over latent dimensions
                )
            )

            # Combine both to get total loss
            total_loss = reconstruction_loss + kl_loss

        # Backpropagation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update tracked metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        # Return metrics for logging
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
