'''
The simplest and most intuitive way to build models. Best for linear stacks of layers.
Use Case: Quick prototyping with feed-forward neural networks or simple architectures.
'''


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


'''
Description:
This script demonstrates building neural network models using the Sequential API in TensorFlow/Keras. The Sequential API is a straightforward way to stack layers sequentially. It is best suited for simple, linear stack architectures like feedforward or convolutional networks.

Behind the Theory:
    - It works well when the flow of data is strictly one direction, without the need for multiple inputs/outputs or non-linear layer connections.
    - Limitations arise when models require advanced connectivity, like shared layers or residual connections.

'''