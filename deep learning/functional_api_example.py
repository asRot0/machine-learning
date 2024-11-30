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
