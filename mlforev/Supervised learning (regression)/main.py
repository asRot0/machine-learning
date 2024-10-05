from data_processing import DataProcessor
from linear_regression_model import LinearRegressionModel
from neural_network_model import NeuralNetworkModel

data_path = "SeoulBikeData.csv"
y_label = "bike_count"
drop_cols = ["wind", "visibility", "functional"]

# Initialize DataProcessor
processor = DataProcessor(data_path, y_label, drop_cols=drop_cols, sample_hour=12)

# View dataset info
processor.data_info()

# Split data
train, val, test = processor.split_data()

# Prepare data for linear regression
X_train, y_train = processor.get_xy(train, x_labels=["temp"])
X_val, y_val = processor.get_xy(val, x_labels=["temp"])
X_test, y_test = processor.get_xy(test, x_labels=["temp"])

# Train and evaluate Linear Regression
lr_model = LinearRegressionModel()
lr_model.fit(X_train, y_train)
score, mse = lr_model.evaluate(X_test, y_test)
print(f"Linear Regression Test Score: {score}, MSE: {mse}")
lr_model.plot_regression(X_train, y_train)


# Train and evaluate Neural Network Model
nn_model = NeuralNetworkModel(input_shape=(1,))
nn_model.train(X_train, y_train, X_val, y_val, epochs=100)
nn_mse = nn_model.evaluate(X_test, y_test)
print(f"Neural Network MSE: {nn_mse}")
nn_model.plot_loss()
nn_model.plot_predictions(X_train, y_train)
