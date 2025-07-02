import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class LatentSpaceVisualizer:
    """
    Utility class to visualize latent space and generated images from a trained VAE model.
    """

    def __init__(self, encoder, decoder, latent_dim=2, output_dir=None):
        """
        Parameters:
        - encoder: Trained VAE encoder model (should output [z_mean, z_log_var, z])
        - decoder: Trained VAE decoder model (takes z and returns generated images)
        - latent_dim: Dimensionality of the latent space (must be 2 for grid)
        - output_dir: Optional directory to save figures
        """
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.output_dir = output_dir

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def plot_latent_scatter(self, x_data, y_labels=None, batch_size=128, title="Latent Space"):
        """
        Projects input data to 2D latent space and shows a scatter plot.
        """
        z_mean, _, _ = self.encoder.predict(x_data, batch_size=batch_size)
        if self.latent_dim != 2:
            raise ValueError("Scatter plot only supported for 2D latent space.")

        plt.figure(figsize=(10, 8))
        if y_labels is not None:
            scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_labels, cmap="tab10", alpha=0.7)
            plt.colorbar(scatter, label="Class Label")
        else:
            plt.scatter(z_mean[:, 0], z_mean[:, 1], alpha=0.5)

        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.title(title)
        plt.grid(True)

        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, "latent_space_scatter.png"))
        plt.show()

    def plot_latent_grid(self, n=15, digit_size=28, z_range=4.0, title="Latent Space Grid"):
        """
        Generates a n x n grid of decoded images from the 2D latent space.
        """
        if self.latent_dim != 2:
            raise ValueError("Latent grid decoding only supported for 2D latent space.")

        figure = np.zeros((digit_size * n, digit_size * n))
        grid_x = np.linspace(-z_range, z_range, n)
        grid_y = np.linspace(-z_range, z_range, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap="gray")
        plt.title(title)
        plt.axis("off")

        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, "latent_grid.png"))
        plt.show()
