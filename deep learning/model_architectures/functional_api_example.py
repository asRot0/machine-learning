'''
Provides flexibility for complex architectures, including multi-input and multi-output models.
Use Case: Architectures with shared layers, skip connections, or multiple outputs.
'''


from tensorflow.keras.layers import Input, Dense, Flatten, concatenate
from tensorflow.keras.models import Model

# Define inputs
input_layer = Input(shape=(28, 28))

# Add layers
x = Flatten()(input_layer)
x1 = Dense(128, activation='relu')(x)
x2 = Dense(64, activation='relu')(x1)

# Create multiple branches
branch1 = Dense(10, activation='softmax', name="branch1")(x2)
branch2 = Dense(10, activation='softmax', name="branch2")(x2)

# Build the model
model = Model(inputs=input_layer, outputs=[branch1, branch2])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


'''
Description:
This script showcases the Functional API for building models. It allows creating complex architectures, including multi-input/multi-output models, shared layers, and directed acyclic graphs of layers.

Behind the Theory:
    - The Functional API introduces flexibility by allowing arbitrary connections between layers.
    - Useful for advanced architectures like ResNet, Inception, and U-Net.
    - Supports layer reuse and branch-based designs, improving efficiency and modularity.

'''