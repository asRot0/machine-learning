'''
Allows for dynamic and completely custom models by subclassing the Model class.
Use Case: Highly dynamic architectures or models requiring custom forward passes.
'''


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten


class MyCustomModel(Model):
    def __init__(self):
        super(MyCustomModel, self).__init__()
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.output_layer = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)


# Build and compile
model = MyCustomModel()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.build(input_shape=(None, 28, 28))  # Specify input shape for summary
model.summary()


'''
Description:
This script introduces the Subclassing API, which is the most flexible way to create dynamic and highly customized models by subclassing the tf.keras.Model class.

Behind the Theory:
    - The Subclassing API allows full control over the forward pass by overriding the call method.
    - Ideal for implementing models that require dynamic behaviors, such as recursive networks or attention mechanisms.
    - However, it requires careful handling of layer connections and is less suited for rapid prototyping compared to Sequential and Functional APIs.

'''