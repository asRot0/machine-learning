# Transfer Learning with Pretrained Model (transfer_learning.py)

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Load pretrained VGG16 model and add new layers for CIFAR-10
base_model = VGG16(include_top=False, input_shape=(32, 32, 3))
model = Sequential(base_model.layers + [Flatten(), Dense(10, activation='softmax')])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
print("Transfer Learning Accuracy:", model.evaluate(X_test, y_test, verbose=0)[1])


'''
Explanation:

Transfer learning script leveraging VGG16 for CIFAR-10 classification, adjusting the output layer to match the dataset classes.
'''