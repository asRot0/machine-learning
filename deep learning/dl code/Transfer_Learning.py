# Using Pretrained CNNs and Fine-tuning

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# Load pretrained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(10, activation='softmax')
])

# Fine-tuning specific layers
for layer in base_model.layers:
    layer.trainable = False  # freeze the layers

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Explanation:
# Transfer learning allows us to leverage pre-trained knowledge from models like VGG16.
# By freezing early layers, we only retrain the final layers to adapt the model to our specific data.
