import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()

# ========== Logistic Regression for Petal Width ==========
# Extract petal width (3rd feature) and target class (Iris-Virginica is class 2)
X_petal_width = iris['data'][:, 3:]  # petal width
y_virginica = (iris['target'] == 2).astype(int)  # Iris-Virginica class

# Fit logistic regression model for petal width
log_reg_pw = LogisticRegression()
log_reg_pw.fit(X_petal_width, y_virginica)

# Predict probabilities
X_new_pw = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba_pw = log_reg_pw.predict_proba(X_new_pw)

# Set figure size
plt.figure(figsize=(10, 6))

# Plot the probabilities
plt.plot(X_new_pw, y_proba_pw[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new_pw, y_proba_pw[:, 0], "b--", label="Not Iris-Virginica")
plt.axvline(x=1.6, color='r', linestyle=':', label="Decision Boundary (1.6 cm)")

# Scatter points
plt.scatter(X_petal_width[y_virginica == 1], y_virginica[y_virginica == 1], marker='^', color='green', label="Iris-Virginica")
plt.scatter(X_petal_width[y_virginica == 0], y_virginica[y_virginica == 0], marker='s', color='blue', label="Not Iris-Virginica")

# Labels and legend
plt.xlabel("Petal width (cm)")
plt.ylabel("Probability")
plt.legend()
plt.title("Logistic Regression - Petal Width for Iris-Virginica")
plt.show()

# ========== Logistic Regression for Petal Length and Width ==========
# Extract petal length (2nd) and petal width (3rd) features
X_petal_len_width = iris['data'][:, 2:4]  # petal length and petal width
y_virginica = (iris['target'] == 2).astype(int)  # Iris-Virginica class

# Fit logistic regression model for petal length and width
log_reg_len_width = LogisticRegression()
log_reg_len_width.fit(X_petal_len_width, y_virginica)

# Create a grid for prediction
x0, x1 = np.meshgrid(np.linspace(1, 7, 500), np.linspace(0, 3, 200))
X_new_len_width = np.c_[x0.ravel(), x1.ravel()]

# Predict probabilities
y_proba_len_width = log_reg_len_width.predict_proba(X_new_len_width)[:, 1].reshape(x0.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(x0, x1, y_proba_len_width, cmap="coolwarm", levels=100, alpha=0.8)

# Scatter original data points
plt.scatter(X_petal_len_width[y_virginica == 1][:, 0], X_petal_len_width[y_virginica == 1][:, 1], color='green', marker='^', label="Iris-Virginica")
plt.scatter(X_petal_len_width[y_virginica == 0][:, 0], X_petal_len_width[y_virginica == 0][:, 1], color='blue', marker='s', label="Not Iris-Virginica")

# Labels and legend
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Logistic Regression Decision Boundary (Petal Length vs. Width)")
plt.legend()
plt.show()

# ========== Softmax Regression for Multiclass Classification ==========
# Use petal length (2nd) and petal width (3rd) for all classes
X_softmax = iris["data"][:, (2, 3)]  # petal length, petal width
y_softmax = iris["target"]  # all classes

# Fit softmax regression
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X_softmax, y_softmax)

# Predict a new sample (5 cm petal length, 2 cm petal width)
sample = np.array([[5, 2]])
class_prediction = softmax_reg.predict(sample)
class_probabilities = softmax_reg.predict_proba(sample)

# Print results
print(f"Predicted class: {class_prediction[0]}")
print(f"Class probabilities: {class_probabilities}")

# Decision boundaries for softmax
x0, x1 = np.meshgrid(np.linspace(1, 7, 200), np.linspace(0, 3, 500))  # Adjust the ranges and shape
# Predict probabilities
X_new_softmax = np.c_[x0.ravel(), x1.ravel()]
y_proba_softmax = softmax_reg.predict_proba(X_new_softmax).reshape(x0.shape[0], x1.shape[1], 3)

# Plot the decision boundaries
plt.figure(figsize=(10, 6))
plt.contourf(x0, x1, y_proba_softmax[:, :, 1], cmap="coolwarm", levels=100, alpha=0.8)

# Scatter original data points
plt.scatter(X_softmax[y_softmax == 0][:, 0], X_softmax[y_softmax == 0][:, 1], color='yellow', marker='o', label="Iris-Setosa")
plt.scatter(X_softmax[y_softmax == 1][:, 0], X_softmax[y_softmax == 1][:, 1], color='blue', marker='s', label="Iris-Versicolor")
plt.scatter(X_softmax[y_softmax == 2][:, 0], X_softmax[y_softmax == 2][:, 1], color='green', marker='^', label="Iris-Virginica")

# Labels and legend
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Softmax Regression Decision Boundaries")
plt.legend()
plt.show()
