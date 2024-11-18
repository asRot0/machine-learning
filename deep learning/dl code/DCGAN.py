# Generative Models
# DCGAN (Deep Convolutional GAN) for Image Generation

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Reshape, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist

# Load and preprocess MNIST data
(X_train, _), (_, _) = mnist.load_data()
X_train = X_train / 127.5 - 1  # normalize images to [-1, 1]
X_train = np.expand_dims(X_train, axis=-1)

# Generator model
def build_generator():
    model = Sequential([
        Dense(128 * 7 * 7, input_shape=(100,)),
        LeakyReLU(0.2),
        Reshape((7, 7, 128)),
        Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),
        Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='tanh')
    ])
    return model

# Discriminator model
def build_discriminator():
    model = Sequential([
        Conv2D(64, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding='same'),
        LeakyReLU(0.2),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

# Explanation:
# DCGAN architecture has two components: a Generator and a Discriminator.
# The Generator creates images from random noise, while the Discriminator attempts to distinguish between real and fake images.
