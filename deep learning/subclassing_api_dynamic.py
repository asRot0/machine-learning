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
