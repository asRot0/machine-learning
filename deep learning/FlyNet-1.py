import tensorflow as tf
from tensorflow import keras

# Define DefaultConv2D with partial
DefaultConv2D = keras.layers.Conv2D


# Define FlyNet layer
class FlyNetBlock(keras.layers.Layer):
    def __init__(self, filters, activation1="relu", activation2="tanh", **kwargs):
        super().__init__(**kwargs)
        self.conv1 = DefaultConv2D(filters, kernel_size=3, padding="same", activation=activation1)
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = DefaultConv2D(filters, kernel_size=3, padding="same", activation=activation2)
        self.bn2 = keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


# Build FlyNet model
model = keras.models.Sequential()

# Initial Conv layer
model.add(DefaultConv2D(32, kernel_size=7, strides=2, padding="same", input_shape=[224, 224, 3], activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))

# Add FlyNet Blocks
model.add(FlyNetBlock(32))
model.add(FlyNetBlock(64))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))
model.add(FlyNetBlock(128))
model.add(FlyNetBlock(256))
model.add(keras.layers.GlobalAveragePooling2D())

# Final Dense layers
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation="softmax"))

# Model summary
model.summary()

# Model Architecture Description
'''
FlyNet Model Architecture:

1. **Initial Conv Layer**:
   - Conv2D layer with 32 filters, kernel size 7x7, stride 2, and padding "same".
   - Batch Normalization and ReLU activation.
   - MaxPooling with pool size 2x2, stride 2.

2. **FlyNet Blocks**:
   - Each block contains two Conv2D layers with Batch Normalization after each layer.
   - The first Conv2D uses "relu" activation, and the second uses "tanh" activation.

   - Block 1: 32 filters.
   - Block 2: 64 filters, followed by MaxPooling with pool size 2x2, stride 2.
   - Block 3: 128 filters.
   - Block 4: 256 filters.

3. **Global Average Pooling and Dense Layers**:
   - Global Average Pooling layer reduces spatial dimensions to 1x1.
   - Flatten layer.
   - Dense layer with 128 units and ReLU activation, followed by a Dropout layer with 50% rate.
   - Dense layer with 10 units and softmax activation for classification.

Model Summary:
The summary provides detailed information on each layer, its output shape, and the number of parameters.
Model: "sequential"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 112, 112, 32)   │         4,736 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 112, 112, 32)   │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 56, 56, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ fly_net_block (FlyNetBlock)     │ (None, 56, 56, 32)     │        18,752 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ fly_net_block_1 (FlyNetBlock)   │ (None, 56, 56, 64)     │        55,936 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 28, 28, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ fly_net_block_2 (FlyNetBlock)   │ (None, 28, 28, 128)    │       222,464 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ fly_net_block_3 (FlyNetBlock)   │ (None, 28, 28, 256)    │       887,296 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d        │ (None, 256)            │             0 │
│ (GlobalAveragePooling2D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 128)            │        32,896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 10)             │         1,290 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 1,223,498 (4.67 MB)
 Trainable params: 1,221,514 (4.66 MB)
 Non-trainable params: 1,984 (7.75 KB)
'''
