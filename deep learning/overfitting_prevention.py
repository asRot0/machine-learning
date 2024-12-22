import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Generate a small synthetic dataset
'''We create a synthetic dataset of 100 samples with 2 features for binary classification.'''
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, size=(100,))

# Split into training and validation sets
X_train, X_val = X[:80], X[80:]
y_train, y_val = y[:80], y[80:]

# Define a helper function to print model summaries
def print_model_summary(title, model):
    print(f"\n{'='*20} {title} {'='*20}")
    model.summary()

# Model with ℓ1 and ℓ2 Regularization
'''
ℓ1 and ℓ2 Regularization:
Regularization penalizes large weights in the network. ℓ1 adds a penalty proportional to the absolute value of weights, 
leading to sparse weights, while ℓ2 penalizes the square of weights, leading to smaller weights overall.
'''
l1_l2_model = Sequential([
    Dense(16, activation='relu', kernel_regularizer=l1(0.01), input_shape=(2,)),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='sigmoid')
])
l1_l2_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print_model_summary("L1 and L2 Regularization Model", l1_l2_model)

# Model with Dropout
'''
Dropout:
Dropout randomly drops a fraction of neurons during training to prevent over-reliance on specific neurons.
This encourages the model to learn more robust features.
'''
dropout_model = Sequential([
    Dense(16, activation='relu', input_shape=(2,)),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
dropout_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print_model_summary("Dropout Model", dropout_model)

# Model with Max-Norm Regularization
'''
Max-Norm Regularization:
Max-Norm regularization constrains the norm of weight vectors to a maximum value, 
ensuring weights do not grow too large during training.
'''
max_norm_model = Sequential([
    Dense(16, activation='relu', kernel_constraint=tf.keras.constraints.max_norm(2.), input_shape=(2,)),
    Dense(16, activation='relu', kernel_constraint=tf.keras.constraints.max_norm(2.)),
    Dense(1, activation='sigmoid')
])
max_norm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print_model_summary("Max-Norm Regularization Model", max_norm_model)

# Monte-Carlo (MC) Dropout Model
'''
Monte-Carlo (MC) Dropout:
MC Dropout applies dropout at inference time to obtain a distribution of predictions. 
This technique can be used to estimate model uncertainty.
'''
mc_dropout_model = Sequential([
    Dense(16, activation='relu', input_shape=(2,)),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
mc_dropout_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# EarlyStopping
'''
EarlyStopping:
This stops training when the validation performance stops improving, avoiding overfitting 
from prolonged training.
'''
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train one model as an example
history = l1_l2_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=[early_stopping_callback],
    verbose=1
)
