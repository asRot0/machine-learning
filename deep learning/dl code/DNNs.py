# Deep Neural Networks (DNNs) with Dropout and Regularization

from tensorflow.keras.layers import Dropout

# Deep Neural Network model with dropout
dnn_model = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile model with regularization
dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dnn_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# Explanation:
# This DNN model applies dropout after each hidden layer to prevent overfitting.
# Dropout randomly "drops" neurons during training to help the model generalize better.
