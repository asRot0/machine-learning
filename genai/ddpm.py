import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.api import layers
import tensorflow_datasets as tfds

# hyperparameters
batch_size = 32
num_epochs = 1
total_timesteps = 1000
norm_groups = 8  # number of groups used in GroupNormalization layer
learning_rate = 2e-4

img_size = 64
img_channels = 3
clip_min = -1.0
clip_max = 1.0

first_conv_channels = 64
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2  # number of residual blocks

dataset_name = None
splits = ["train"]
train_ds = None
