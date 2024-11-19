# Custom Gradient Descent, Adam, RMSProp Implementations in TensorFlow

from tensorflow.keras.optimizers import SGD, Adam, RMSprop

# Compile model with different optimizers for comparison
model.compile(optimizer=SGD(learning_rate=0.01), loss='mse', metrics=['accuracy'])
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['accuracy'])
model.compile(optimizer=RMSprop(learning_rate=0.01), loss='mse', metrics=['accuracy'])

# Explanation:
# Gradient Descent: iteratively adjusts weights to minimize error using gradients.
# Adam and RMSProp: more advanced algorithms, adapting learning rate for each parameter.
