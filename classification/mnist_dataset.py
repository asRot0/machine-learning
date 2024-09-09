from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mnist = fetch_openml('mnist_784', version=1)

print(type(mnist))
print(mnist.keys())

X, y = mnist['data'], mnist['target']
print(X.shape, y.shape)
print(type(X))
print(type(y))
y = y.astype(np.uint8)

'''
some_digit = X.iloc[0].to_numpy()
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap=mpl.colormaps['PuBu'], interpolation='nearest')
plt.axis('off')
plt.show()

'''

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = y_train == 5
y_test_5 = y_test == 5

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

while True:
    try:
        index = int(input('Enter digit index (0-69999): '))
        if index < 0 or index >= len(X):
            print("Index out of range. Please enter a number between 0 and 69999.")
            continue

        some_digit = X.iloc[index].to_numpy()  # Convert DataFrame row to NumPy array
        print(sgd_clf.predict([some_digit]))

    except ValueError:
        print("Invalid input. Please enter a valid integer.")
    except KeyboardInterrupt:
        print("\nExiting.")
        break
