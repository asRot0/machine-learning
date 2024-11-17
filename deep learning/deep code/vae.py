# Variational Autoencoder (VAE) (vae.py)

from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
import numpy as np

# Encoder and Decoder model
latent_dim = 2
inputs = Input(shape=(784,))
h = Dense(256, activation='relu')(inputs)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder model
decoder_h = Dense(256, activation='relu')
decoder_out = Dense(784, activation='sigmoid')
h_decoded = decoder_h(z)
outputs = decoder_out(h_decoded)

# Full VAE model
vae = Model(inputs, outputs)
vae.compile(optimizer='adam', loss='binary_crossentropy')

# Example data to train on
X_train = np.random.rand(1000, 784)  # Replace with actual data
vae.fit(X_train, X_train, epochs=10, batch_size=32)


'''
Explanation:

Variational Autoencoder that learns a compressed representation of data.
Implements sampling from latent space, ideal for generating new samples similar to the input data.
'''