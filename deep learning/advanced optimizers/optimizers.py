'''
Script: momentum_optimizer.py
Description: Implements Momentum Optimization for neural network training.
Momentum accelerates gradient descent by adding a fraction of the previous update to the current update.
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(1000, 20)
y = np.random.randint(0, 2, 1000)

# Create model with Momentum Optimization
def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(20,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    optimizer = SGD(learning_rate=0.01, momentum=0.9)  # SGD with momentum
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train model
model = create_model()
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

'''
Script: nesterov_optimizer.py
Description: Implements Nesterov Accelerated Gradient (NAG) optimization. 
NAG computes gradients at a lookahead position to achieve faster convergence.
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(1000, 20)
y = np.random.randint(0, 2, 1000)

# Create model with Nesterov Accelerated Gradient
def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(20,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)  # SGD with Nesterov momentum
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train model
model = create_model()
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

'''
Script: adagrad_optimizer.py
Description: Implements AdaGrad optimization. 
AdaGrad adapts learning rates based on historical gradient information, making it suitable for sparse data.
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adagrad
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(1000, 20)
y = np.random.randint(0, 2, 1000)

# Create model with AdaGrad optimizer
def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(20,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adagrad(learning_rate=0.01)  # AdaGrad optimizer
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train model
model = create_model()
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

'''
Script: rmsprop_optimizer.py
Description: Implements RMSProp optimization. 
RMSProp is designed to work well in non-stationary environments and is widely used for RNNs.
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(1000, 20)
y = np.random.randint(0, 2, 1000)

# Create model with RMSProp optimizer
def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(20,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    optimizer = RMSprop(learning_rate=0.001)  # RMSProp optimizer
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train model
model = create_model()
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

'''
Script: adam_nadam_optimizer.py
Description: Implements Adam and Nadam optimizers. 
Adam combines momentum and RMSProp, while Nadam adds Nesterov momentum to Adam.
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, Nadam
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(1000, 20)
y = np.random.randint(0, 2, 1000)

# Create model with Adam optimizer
def create_adam_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(20,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=0.001)  # Adam optimizer
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create model with Nadam optimizer
def create_nadam_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(20,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Nadam(learning_rate=0.001)  # Nadam optimizer
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train models
adam_model = create_adam_model()
adam_history = adam_model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

nadam_model = create_nadam_model()
nadam_history = nadam_model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

'''
Script: learning_rate_schedule.py
Description: Demonstrates learning rate scheduling methods including step decay, time-based decay, and exponential decay.
'''

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(1000, 20)
y = np.random.randint(0, 2, 1000)

# Step decay function
def step_decay(epoch):
    initial_lr = 0.1
    drop = 0.5
    epochs_drop = 5
    return initial_lr * (drop ** (epoch // epochs_drop))

# Create model with step decay learning rate schedule
def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(20,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Apply learning rate scheduler
model = create_model()
lr_scheduler = LearningRateScheduler(step_decay)
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=[lr_scheduler])
