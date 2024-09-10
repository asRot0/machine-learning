from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
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


def pltDigit(index):
    some_digit = X.iloc[index].to_numpy()
    some_digit_image = some_digit.reshape(28, 28)

    plt.imshow(some_digit_image, cmap=mpl.colormaps['PuBu'], interpolation='nearest')
    plt.axis('off')
    plt.show()


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = y_train == 5
y_test_5 = y_test == 5

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

'''
while True:
    try:
        index = int(input('Enter digit index (0-69999): '))
        if index < 0 or index >= len(X):
            print("Index out of range")
            break

        some_digit = X.iloc[index]
        print(sgd_clf.predict([some_digit]))

        pltDigit(index)

    except ValueError:
        print("Invalid input. Please enter a valid integer.")
    except KeyboardInterrupt:
        print("\nExiting.")
        break
'''
some_digit = X.iloc[0]
print(sgd_clf.predict([some_digit]))

cross_acuuracy = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
print(cross_acuuracy)

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

print(confusion_matrix(y_train_5, y_train_pred))