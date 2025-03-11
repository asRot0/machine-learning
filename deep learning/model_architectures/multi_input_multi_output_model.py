import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Concatenate
from tensorflow.keras.models import Model

# Define Input Layers
numerical_input = Input(shape=(10,), name="numerical_input")  # Example: 10 numerical features
image_input = Input(shape=(32, 32, 3), name="image_input")    # Example: 32x32 RGB image

# Process Numerical Input
x1 = Dense(32, activation="relu")(numerical_input)
x1 = Dense(16, activation="relu")(x1)

# Process Image Input
x2 = Conv2D(32, (3,3), activation="relu")(image_input)
x2 = Flatten()(x2)
x2 = Dense(64, activation="relu")(x2)

# Merge Both Feature Representations
merged = Concatenate()([x1, x2])

# Define Output Branches
classification_output = Dense(3, activation="softmax", name="classification_output")(merged)  # 3 classes
regression_output = Dense(1, activation="linear", name="regression_output")(merged)  # Continuous value

# Define Model
model = Model(inputs=[numerical_input, image_input], outputs=[classification_output, regression_output])

# Compile Model
model.compile(optimizer="adam",
              loss={"classification_output": "categorical_crossentropy",
                    "regression_output": "mse"},
              metrics={"classification_output": "accuracy",
                       "regression_output": "mae"})

# Summary
model.summary()

'''
Two Different Inputs

numerical_input: Processes structured data using dense layers.
image_input: Uses Conv2D and Dense layers to process image data.

Feature Extraction

Both inputs are transformed separately before merging.

Concatenation

Merging different input representations using Concatenate().

Two Different Outputs

classification_output: Uses softmax for multi-class classification.
regression_output: Uses linear activation for continuous output.

Compilation

Multiple losses (categorical_crossentropy and mse).
Multiple metrics (accuracy and mae).
'''


num_samples = 100
numerical_data = np.random.rand(num_samples, 10)  # Random structured data
image_data = np.random.rand(num_samples, 32, 32, 3)  # Random images
classification_labels = tf.keras.utils.to_categorical(np.random.randint(3, size=num_samples), 3)
regression_labels = np.random.rand(num_samples, 1)

# Train the model
model.fit([numerical_data, image_data],
          [classification_labels, regression_labels],
          epochs=10, batch_size=16)

'''
Model: "functional"
┌─────────────────────┬───────────────────┬────────────┬───────────────────┐
│ Layer (type)        │ Output Shape      │    Param # │ Connected to      │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ image_input         │ (None, 32, 32, 3) │          0 │ -                 │
│ (InputLayer)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ numerical_input     │ (None, 10)        │          0 │ -                 │
│ (InputLayer)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d (Conv2D)     │ (None, 30, 30,    │        896 │ image_input[0][0] │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense (Dense)       │ (None, 32)        │        352 │ numerical_input[… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ flatten (Flatten)   │ (None, 28800)     │          0 │ conv2d[0][0]      │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_1 (Dense)     │ (None, 16)        │        528 │ dense[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_2 (Dense)     │ (None, 64)        │  1,843,264 │ flatten[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate         │ (None, 80)        │          0 │ dense_1[0][0],    │
│ (Concatenate)       │                   │            │ dense_2[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ classification_out… │ (None, 3)         │        243 │ concatenate[0][0] │
│ (Dense)             │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ regression_output   │ (None, 1)         │         81 │ concatenate[0][0] │
│ (Dense)             │                   │            │                   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 1,845,364 (7.04 MB)
 Trainable params: 1,845,364 (7.04 MB)
 Non-trainable params: 0 (0.00 B)
Epoch 1/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 2s 17ms/step - classification_output_accuracy: 0.3720 - classification_output_loss: 1.8397 - loss: 22.0379 - regression_output_loss: 19.8083 - regression_output_mae: 2.8177
Epoch 2/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - classification_output_accuracy: 0.3150 - classification_output_loss: 1.3812 - loss: 1.6502 - regression_output_loss: 0.2738 - regression_output_mae: 0.4065
Epoch 3/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - classification_output_accuracy: 0.5048 - classification_output_loss: 1.0530 - loss: 1.2866 - regression_output_loss: 0.2358 - regression_output_mae: 0.4019
Epoch 4/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - classification_output_accuracy: 0.5151 - classification_output_loss: 1.0413 - loss: 1.1803 - regression_output_loss: 0.1429 - regression_output_mae: 0.3227
Epoch 5/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - classification_output_accuracy: 0.5580 - classification_output_loss: 1.0050 - loss: 1.1214 - regression_output_loss: 0.1139 - regression_output_mae: 0.2953
Epoch 6/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - classification_output_accuracy: 0.6923 - classification_output_loss: 0.9583 - loss: 1.0415 - regression_output_loss: 0.0786 - regression_output_mae: 0.2285
Epoch 7/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - classification_output_accuracy: 0.6362 - classification_output_loss: 0.9417 - loss: 1.0155 - regression_output_loss: 0.0692 - regression_output_mae: 0.2232
Epoch 8/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - classification_output_accuracy: 0.6693 - classification_output_loss: 0.8625 - loss: 0.9298 - regression_output_loss: 0.0694 - regression_output_mae: 0.2255
Epoch 9/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - classification_output_accuracy: 0.9435 - classification_output_loss: 0.8028 - loss: 0.8557 - regression_output_loss: 0.0544 - regression_output_mae: 0.1804
Epoch 10/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - classification_output_accuracy: 0.9229 - classification_output_loss: 0.7096 - loss: 0.7492 - regression_output_loss: 0.0399 - regression_output_mae: 0.1646

Process finished with exit code 0
'''