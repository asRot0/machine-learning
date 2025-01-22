import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load a pretrained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Display the model summary
model.summary()

# Load and preprocess an example image
img_path = tf.keras.utils.get_file(
    "elephant.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg"
)

# Load the image
img = image.load_img(img_path, target_size=(224, 224))
plt.imshow(img)
plt.title("Input Image")
plt.axis("off")
plt.show()

# Convert the image to an array and preprocess it for MobileNetV2
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = preprocess_input(img_array)

# Predict the image's class
predictions = model.predict(img_array)
decoded_predictions = decode_predictions(predictions, top=3)

# Display predictions
print("\nTop Predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
    print(f"{i + 1}: {label} ({score:.2f})")

'''
2025-01-22 21:41:00.371123: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-01-22 21:41:08.317469: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-01-22 21:41:28.062604: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5
14536120/14536120 ━━━━━━━━━━━━━━━━━━━━ 39s 3us/step
Model: "mobilenetv2_1.00_224"
┌─────────────────────┬───────────────────┬────────────┬───────────────────┐
│ Layer (type)        │ Output Shape      │    Param # │ Connected to      │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ input_layer         │ (None, 224, 224,  │          0 │ -                 │
│ (InputLayer)        │ 3)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ Conv1 (Conv2D)      │ (None, 112, 112,  │        864 │ input_layer[0][0] │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ bn_Conv1            │ (None, 112, 112,  │        128 │ Conv1[0][0]       │
│ (BatchNormalizatio… │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ Conv1_relu (ReLU)   │ (None, 112, 112,  │          0 │ bn_Conv1[0][0]    │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ expanded_conv_dept… │ (None, 112, 112,  │        288 │ Conv1_relu[0][0]  │
│ (DepthwiseConv2D)   │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ expanded_conv_dept… │ (None, 112, 112,  │        128 │ expanded_conv_de… │
│ (BatchNormalizatio… │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ expanded_conv_dept… │ (None, 112, 112,  │          0 │ expanded_conv_de… │
│ (ReLU)              │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ expanded_conv_proj… │ (None, 112, 112,  │        512 │ expanded_conv_de… │
│ (Conv2D)            │ 16)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ expanded_conv_proj… │ (None, 112, 112,  │         64 │ expanded_conv_pr… │
│ (BatchNormalizatio… │ 16)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_expand      │ (None, 112, 112,  │      1,536 │ expanded_conv_pr… │
│ (Conv2D)            │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_expand_BN   │ (None, 112, 112,  │        384 │ block_1_expand[0… │
│ (BatchNormalizatio… │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_expand_relu │ (None, 112, 112,  │          0 │ block_1_expand_B… │
│ (ReLU)              │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_pad         │ (None, 113, 113,  │          0 │ block_1_expand_r… │
│ (ZeroPadding2D)     │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_depthwise   │ (None, 56, 56,    │        864 │ block_1_pad[0][0] │
│ (DepthwiseConv2D)   │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_depthwise_… │ (None, 56, 56,    │        384 │ block_1_depthwis… │
│ (BatchNormalizatio… │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_depthwise_… │ (None, 56, 56,    │          0 │ block_1_depthwis… │
│ (ReLU)              │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_project     │ (None, 56, 56,    │      2,304 │ block_1_depthwis… │
│ (Conv2D)            │ 24)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_1_project_BN  │ (None, 56, 56,    │         96 │ block_1_project[… │
│ (BatchNormalizatio… │ 24)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_expand      │ (None, 56, 56,    │      3,456 │ block_1_project_… │
│ (Conv2D)            │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_expand_BN   │ (None, 56, 56,    │        576 │ block_2_expand[0… │
│ (BatchNormalizatio… │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_expand_relu │ (None, 56, 56,    │          0 │ block_2_expand_B… │
│ (ReLU)              │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_depthwise   │ (None, 56, 56,    │      1,296 │ block_2_expand_r… │
│ (DepthwiseConv2D)   │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_depthwise_… │ (None, 56, 56,    │        576 │ block_2_depthwis… │
│ (BatchNormalizatio… │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_depthwise_… │ (None, 56, 56,    │          0 │ block_2_depthwis… │
│ (ReLU)              │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_project     │ (None, 56, 56,    │      3,456 │ block_2_depthwis… │
│ (Conv2D)            │ 24)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_project_BN  │ (None, 56, 56,    │         96 │ block_2_project[… │
│ (BatchNormalizatio… │ 24)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_2_add (Add)   │ (None, 56, 56,    │          0 │ block_1_project_… │
│                     │ 24)               │            │ block_2_project_… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_expand      │ (None, 56, 56,    │      3,456 │ block_2_add[0][0] │
│ (Conv2D)            │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_expand_BN   │ (None, 56, 56,    │        576 │ block_3_expand[0… │
│ (BatchNormalizatio… │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_expand_relu │ (None, 56, 56,    │          0 │ block_3_expand_B… │
│ (ReLU)              │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_pad         │ (None, 57, 57,    │          0 │ block_3_expand_r… │
│ (ZeroPadding2D)     │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_depthwise   │ (None, 28, 28,    │      1,296 │ block_3_pad[0][0] │
│ (DepthwiseConv2D)   │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_depthwise_… │ (None, 28, 28,    │        576 │ block_3_depthwis… │
│ (BatchNormalizatio… │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_depthwise_… │ (None, 28, 28,    │          0 │ block_3_depthwis… │
│ (ReLU)              │ 144)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_project     │ (None, 28, 28,    │      4,608 │ block_3_depthwis… │
│ (Conv2D)            │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_3_project_BN  │ (None, 28, 28,    │        128 │ block_3_project[… │
│ (BatchNormalizatio… │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_expand      │ (None, 28, 28,    │      6,144 │ block_3_project_… │
│ (Conv2D)            │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_expand_BN   │ (None, 28, 28,    │        768 │ block_4_expand[0… │
│ (BatchNormalizatio… │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_expand_relu │ (None, 28, 28,    │          0 │ block_4_expand_B… │
│ (ReLU)              │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_depthwise   │ (None, 28, 28,    │      1,728 │ block_4_expand_r… │
│ (DepthwiseConv2D)   │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_depthwise_… │ (None, 28, 28,    │        768 │ block_4_depthwis… │
│ (BatchNormalizatio… │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_depthwise_… │ (None, 28, 28,    │          0 │ block_4_depthwis… │
│ (ReLU)              │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_project     │ (None, 28, 28,    │      6,144 │ block_4_depthwis… │
│ (Conv2D)            │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_project_BN  │ (None, 28, 28,    │        128 │ block_4_project[… │
│ (BatchNormalizatio… │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_4_add (Add)   │ (None, 28, 28,    │          0 │ block_3_project_… │
│                     │ 32)               │            │ block_4_project_… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_expand      │ (None, 28, 28,    │      6,144 │ block_4_add[0][0] │
│ (Conv2D)            │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_expand_BN   │ (None, 28, 28,    │        768 │ block_5_expand[0… │
│ (BatchNormalizatio… │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_expand_relu │ (None, 28, 28,    │          0 │ block_5_expand_B… │
│ (ReLU)              │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_depthwise   │ (None, 28, 28,    │      1,728 │ block_5_expand_r… │
│ (DepthwiseConv2D)   │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_depthwise_… │ (None, 28, 28,    │        768 │ block_5_depthwis… │
│ (BatchNormalizatio… │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_depthwise_… │ (None, 28, 28,    │          0 │ block_5_depthwis… │
│ (ReLU)              │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_project     │ (None, 28, 28,    │      6,144 │ block_5_depthwis… │
│ (Conv2D)            │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_project_BN  │ (None, 28, 28,    │        128 │ block_5_project[… │
│ (BatchNormalizatio… │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_5_add (Add)   │ (None, 28, 28,    │          0 │ block_4_add[0][0… │
│                     │ 32)               │            │ block_5_project_… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_expand      │ (None, 28, 28,    │      6,144 │ block_5_add[0][0] │
│ (Conv2D)            │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_expand_BN   │ (None, 28, 28,    │        768 │ block_6_expand[0… │
│ (BatchNormalizatio… │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_expand_relu │ (None, 28, 28,    │          0 │ block_6_expand_B… │
│ (ReLU)              │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_pad         │ (None, 29, 29,    │          0 │ block_6_expand_r… │
│ (ZeroPadding2D)     │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_depthwise   │ (None, 14, 14,    │      1,728 │ block_6_pad[0][0] │
│ (DepthwiseConv2D)   │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_depthwise_… │ (None, 14, 14,    │        768 │ block_6_depthwis… │
│ (BatchNormalizatio… │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_depthwise_… │ (None, 14, 14,    │          0 │ block_6_depthwis… │
│ (ReLU)              │ 192)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_project     │ (None, 14, 14,    │     12,288 │ block_6_depthwis… │
│ (Conv2D)            │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_6_project_BN  │ (None, 14, 14,    │        256 │ block_6_project[… │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_expand      │ (None, 14, 14,    │     24,576 │ block_6_project_… │
│ (Conv2D)            │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_expand_BN   │ (None, 14, 14,    │      1,536 │ block_7_expand[0… │
│ (BatchNormalizatio… │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_expand_relu │ (None, 14, 14,    │          0 │ block_7_expand_B… │
│ (ReLU)              │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_depthwise   │ (None, 14, 14,    │      3,456 │ block_7_expand_r… │
│ (DepthwiseConv2D)   │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_depthwise_… │ (None, 14, 14,    │      1,536 │ block_7_depthwis… │
│ (BatchNormalizatio… │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_depthwise_… │ (None, 14, 14,    │          0 │ block_7_depthwis… │
│ (ReLU)              │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_project     │ (None, 14, 14,    │     24,576 │ block_7_depthwis… │
│ (Conv2D)            │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_project_BN  │ (None, 14, 14,    │        256 │ block_7_project[… │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_7_add (Add)   │ (None, 14, 14,    │          0 │ block_6_project_… │
│                     │ 64)               │            │ block_7_project_… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_expand      │ (None, 14, 14,    │     24,576 │ block_7_add[0][0] │
│ (Conv2D)            │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_expand_BN   │ (None, 14, 14,    │      1,536 │ block_8_expand[0… │
│ (BatchNormalizatio… │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_expand_relu │ (None, 14, 14,    │          0 │ block_8_expand_B… │
│ (ReLU)              │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_depthwise   │ (None, 14, 14,    │      3,456 │ block_8_expand_r… │
│ (DepthwiseConv2D)   │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_depthwise_… │ (None, 14, 14,    │      1,536 │ block_8_depthwis… │
│ (BatchNormalizatio… │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_depthwise_… │ (None, 14, 14,    │          0 │ block_8_depthwis… │
│ (ReLU)              │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_project     │ (None, 14, 14,    │     24,576 │ block_8_depthwis… │
│ (Conv2D)            │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_project_BN  │ (None, 14, 14,    │        256 │ block_8_project[… │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_8_add (Add)   │ (None, 14, 14,    │          0 │ block_7_add[0][0… │
│                     │ 64)               │            │ block_8_project_… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_expand      │ (None, 14, 14,    │     24,576 │ block_8_add[0][0] │
│ (Conv2D)            │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_expand_BN   │ (None, 14, 14,    │      1,536 │ block_9_expand[0… │
│ (BatchNormalizatio… │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_expand_relu │ (None, 14, 14,    │          0 │ block_9_expand_B… │
│ (ReLU)              │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_depthwise   │ (None, 14, 14,    │      3,456 │ block_9_expand_r… │
│ (DepthwiseConv2D)   │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_depthwise_… │ (None, 14, 14,    │      1,536 │ block_9_depthwis… │
│ (BatchNormalizatio… │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_depthwise_… │ (None, 14, 14,    │          0 │ block_9_depthwis… │
│ (ReLU)              │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_project     │ (None, 14, 14,    │     24,576 │ block_9_depthwis… │
│ (Conv2D)            │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_project_BN  │ (None, 14, 14,    │        256 │ block_9_project[… │
│ (BatchNormalizatio… │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_9_add (Add)   │ (None, 14, 14,    │          0 │ block_8_add[0][0… │
│                     │ 64)               │            │ block_9_project_… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_10_expand     │ (None, 14, 14,    │     24,576 │ block_9_add[0][0] │
│ (Conv2D)            │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_10_expand_BN  │ (None, 14, 14,    │      1,536 │ block_10_expand[… │
│ (BatchNormalizatio… │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_10_expand_re… │ (None, 14, 14,    │          0 │ block_10_expand_… │
│ (ReLU)              │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_10_depthwise  │ (None, 14, 14,    │      3,456 │ block_10_expand_… │
│ (DepthwiseConv2D)   │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_10_depthwise… │ (None, 14, 14,    │      1,536 │ block_10_depthwi… │
│ (BatchNormalizatio… │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_10_depthwise… │ (None, 14, 14,    │          0 │ block_10_depthwi… │
│ (ReLU)              │ 384)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_10_project    │ (None, 14, 14,    │     36,864 │ block_10_depthwi… │
│ (Conv2D)            │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_10_project_BN │ (None, 14, 14,    │        384 │ block_10_project… │
│ (BatchNormalizatio… │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_expand     │ (None, 14, 14,    │     55,296 │ block_10_project… │
│ (Conv2D)            │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_expand_BN  │ (None, 14, 14,    │      2,304 │ block_11_expand[… │
│ (BatchNormalizatio… │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_expand_re… │ (None, 14, 14,    │          0 │ block_11_expand_… │
│ (ReLU)              │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_depthwise  │ (None, 14, 14,    │      5,184 │ block_11_expand_… │
│ (DepthwiseConv2D)   │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_depthwise… │ (None, 14, 14,    │      2,304 │ block_11_depthwi… │
│ (BatchNormalizatio… │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_depthwise… │ (None, 14, 14,    │          0 │ block_11_depthwi… │
│ (ReLU)              │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_project    │ (None, 14, 14,    │     55,296 │ block_11_depthwi… │
│ (Conv2D)            │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_project_BN │ (None, 14, 14,    │        384 │ block_11_project… │
│ (BatchNormalizatio… │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_11_add (Add)  │ (None, 14, 14,    │          0 │ block_10_project… │
│                     │ 96)               │            │ block_11_project… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_expand     │ (None, 14, 14,    │     55,296 │ block_11_add[0][… │
│ (Conv2D)            │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_expand_BN  │ (None, 14, 14,    │      2,304 │ block_12_expand[… │
│ (BatchNormalizatio… │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_expand_re… │ (None, 14, 14,    │          0 │ block_12_expand_… │
│ (ReLU)              │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_depthwise  │ (None, 14, 14,    │      5,184 │ block_12_expand_… │
│ (DepthwiseConv2D)   │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_depthwise… │ (None, 14, 14,    │      2,304 │ block_12_depthwi… │
│ (BatchNormalizatio… │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_depthwise… │ (None, 14, 14,    │          0 │ block_12_depthwi… │
│ (ReLU)              │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_project    │ (None, 14, 14,    │     55,296 │ block_12_depthwi… │
│ (Conv2D)            │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_project_BN │ (None, 14, 14,    │        384 │ block_12_project… │
│ (BatchNormalizatio… │ 96)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_12_add (Add)  │ (None, 14, 14,    │          0 │ block_11_add[0][… │
│                     │ 96)               │            │ block_12_project… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_expand     │ (None, 14, 14,    │     55,296 │ block_12_add[0][… │
│ (Conv2D)            │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_expand_BN  │ (None, 14, 14,    │      2,304 │ block_13_expand[… │
│ (BatchNormalizatio… │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_expand_re… │ (None, 14, 14,    │          0 │ block_13_expand_… │
│ (ReLU)              │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_pad        │ (None, 15, 15,    │          0 │ block_13_expand_… │
│ (ZeroPadding2D)     │ 576)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_depthwise  │ (None, 7, 7, 576) │      5,184 │ block_13_pad[0][… │
│ (DepthwiseConv2D)   │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_depthwise… │ (None, 7, 7, 576) │      2,304 │ block_13_depthwi… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_depthwise… │ (None, 7, 7, 576) │          0 │ block_13_depthwi… │
│ (ReLU)              │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_project    │ (None, 7, 7, 160) │     92,160 │ block_13_depthwi… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_13_project_BN │ (None, 7, 7, 160) │        640 │ block_13_project… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_expand     │ (None, 7, 7, 960) │    153,600 │ block_13_project… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_expand_BN  │ (None, 7, 7, 960) │      3,840 │ block_14_expand[… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_expand_re… │ (None, 7, 7, 960) │          0 │ block_14_expand_… │
│ (ReLU)              │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_depthwise  │ (None, 7, 7, 960) │      8,640 │ block_14_expand_… │
│ (DepthwiseConv2D)   │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_depthwise… │ (None, 7, 7, 960) │      3,840 │ block_14_depthwi… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_depthwise… │ (None, 7, 7, 960) │          0 │ block_14_depthwi… │
│ (ReLU)              │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_project    │ (None, 7, 7, 160) │    153,600 │ block_14_depthwi… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_project_BN │ (None, 7, 7, 160) │        640 │ block_14_project… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_14_add (Add)  │ (None, 7, 7, 160) │          0 │ block_13_project… │
│                     │                   │            │ block_14_project… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_expand     │ (None, 7, 7, 960) │    153,600 │ block_14_add[0][… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_expand_BN  │ (None, 7, 7, 960) │      3,840 │ block_15_expand[… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_expand_re… │ (None, 7, 7, 960) │          0 │ block_15_expand_… │
│ (ReLU)              │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_depthwise  │ (None, 7, 7, 960) │      8,640 │ block_15_expand_… │
│ (DepthwiseConv2D)   │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_depthwise… │ (None, 7, 7, 960) │      3,840 │ block_15_depthwi… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_depthwise… │ (None, 7, 7, 960) │          0 │ block_15_depthwi… │
│ (ReLU)              │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_project    │ (None, 7, 7, 160) │    153,600 │ block_15_depthwi… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_project_BN │ (None, 7, 7, 160) │        640 │ block_15_project… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_15_add (Add)  │ (None, 7, 7, 160) │          0 │ block_14_add[0][… │
│                     │                   │            │ block_15_project… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_16_expand     │ (None, 7, 7, 960) │    153,600 │ block_15_add[0][… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_16_expand_BN  │ (None, 7, 7, 960) │      3,840 │ block_16_expand[… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_16_expand_re… │ (None, 7, 7, 960) │          0 │ block_16_expand_… │
│ (ReLU)              │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_16_depthwise  │ (None, 7, 7, 960) │      8,640 │ block_16_expand_… │
│ (DepthwiseConv2D)   │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_16_depthwise… │ (None, 7, 7, 960) │      3,840 │ block_16_depthwi… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_16_depthwise… │ (None, 7, 7, 960) │          0 │ block_16_depthwi… │
│ (ReLU)              │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_16_project    │ (None, 7, 7, 320) │    307,200 │ block_16_depthwi… │
│ (Conv2D)            │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ block_16_project_BN │ (None, 7, 7, 320) │      1,280 │ block_16_project… │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ Conv_1 (Conv2D)     │ (None, 7, 7,      │    409,600 │ block_16_project… │
│                     │ 1280)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ Conv_1_bn           │ (None, 7, 7,      │      5,120 │ Conv_1[0][0]      │
│ (BatchNormalizatio… │ 1280)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ out_relu (ReLU)     │ (None, 7, 7,      │          0 │ Conv_1_bn[0][0]   │
│                     │ 1280)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ global_average_poo… │ (None, 1280)      │          0 │ out_relu[0][0]    │
│ (GlobalAveragePool… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ predictions (Dense) │ (None, 1000)      │  1,281,000 │ global_average_p… │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 3,538,984 (13.50 MB)
 Trainable params: 3,504,872 (13.37 MB)
 Non-trainable params: 34,112 (133.25 KB)
Downloading data from https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg
4783815/4783815 ━━━━━━━━━━━━━━━━━━━━ 11s 2us/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step
Downloading data from https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
35363/35363 ━━━━━━━━━━━━━━━━━━━━ 0s 1us/step

Top Predictions:
1: African_elephant (0.52)
2: tusker (0.17)
3: Indian_elephant (0.16)

'''