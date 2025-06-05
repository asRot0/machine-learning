import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Discriminator(keras.Model):
    def __init__(self):
        super().__init__(name="discriminator")

        self.model = keras.Sequential([
            layers.Input(shape=(64, 64, 3)),

            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),

            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),

            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),

            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid")
        ])

    def call(self, inputs):
        return self.model(inputs)


class Generator(keras.Model):
    def __init__(self, latent_dim=128):
        super().__init__(name="generator")
        self.latent_dim = latent_dim

        self.model = keras.Sequential([
            layers.Dense(8 * 8 * 128, input_shape=(latent_dim,)),
            layers.Reshape((8, 8, 128)),

            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),

            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),

            layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),

            layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")
        ])

    def call(self, inputs):
        return self.model(inputs)


class GAN(keras.Model):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        self.latent_dim = latent_dim

        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")

        self.cross_entropy = keras.losses.BinaryCrossentropy()

    def compile(self, d_optimizer, g_optimizer):
        super(GAN, self).compile()
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


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=4, latent_dim=128, output_dir="generated_images"):
        super().__init__()
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.seed = tf.random.normal(shape=(num_img, latent_dim))
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generator(self.seed)
        generated_images *= 255.0
        generated_images = tf.clip_by_value(generated_images, 0, 255)
        generated_images = tf.cast(generated_images, tf.uint8).numpy()

        for i in range(self.num_img):
            img = keras.utils.array_to_img(generated_images[i])
            img.save(os.path.join(self.output_dir, f"generated_img_{epoch:03d}_{i}.png"))
