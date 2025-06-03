import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Generator(keras.Model):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.dense1 = layers.Dense(128, activation="relu")
        self.dense2 = layers.Dense(784, activation="sigmoid")  # 28*28

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


class Discriminator(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(128, activation="relu")
        self.dense2 = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


class GAN(keras.Model):
    def __init__(self, generator=None, discriminator=None, latent_dim=100):
        super().__init__()
        self.generator = generator if generator else Generator()
        self.discriminator = discriminator if discriminator else Discriminator()
        self.latent_dim = latent_dim

        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")

        self.cross_entropy = keras.losses.BinaryCrossentropy()

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Generate fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine real and fake images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Labels for fake and real images
        labels = tf.concat([
            tf.zeros((batch_size, 1)),
            tf.ones((batch_size, 1))
        ], axis=0)

        # Add noise to the labels
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.cross_entropy(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Sample random points in latent space again
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Labels for misleading the discriminator (all real)
        misleading_labels = tf.ones((batch_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors)
            predictions = self.discriminator(fake_images)
            g_loss = self.cross_entropy(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Track loss
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result()
        }
