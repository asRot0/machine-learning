# Simple GAN Model (simple_gan.py)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Discriminator and Generator for GAN
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Training the GAN
discriminator = Discriminator()
generator = Generator()
loss_fn = nn.BCELoss()
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Data for training
data_loader = DataLoader(datasets.MNIST('.', transform=transforms.ToTensor(), download=True),
                         batch_size=64, shuffle=True)

# Main Training Loop
for epoch in range(10):
    for real_imgs, _ in data_loader:
        real_imgs = real_imgs.view(-1, 784)
        batch_size = real_imgs.size(0)

        # Train Discriminator
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        d_optimizer.zero_grad()
        real_loss = loss_fn(discriminator(real_imgs), real_labels)
        fake_imgs = generator(torch.randn(batch_size, 100))
        fake_loss = loss_fn(discriminator(fake_imgs.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        gen_loss = loss_fn(discriminator(fake_imgs), real_labels)
        gen_loss.backward()
        g_optimizer.step()


'''
Explanation of the Code:

Discriminator and Generator: The Discriminator distinguishes real images from generated ones, and the Generator creates images from random noise.
Training Loop: The Discriminator is trained to distinguish real images from generated ones, while the Generator tries to improve so that the Discriminator fails.
'''