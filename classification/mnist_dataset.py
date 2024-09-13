from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
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


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='recall')

    plt.xlabel('threshold')
    plt.ylabel('precision/recall')
    plt.legend()
    plt.axis('off')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    # Highlight the threshold (example: threshold = 0.5)
    threshold = 0.5
    plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold = 0.5')

    # Highlight specific precision and recall at that threshold
    precision_at_threshold = precisions[list(thresholds).index(threshold)]
    recall_at_threshold = recalls[list(thresholds).index(threshold)]

    plt.plot(threshold, precision_at_threshold, 'bo')  # Blue dot for precision
    plt.plot(threshold, recall_at_threshold, 'go')  # Green dot for recall


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

cross_acuuracy = cross_val_score(sgd_clf, X_train, y_train_5, cv=3,
                                 scoring='accuracy')
print(cross_acuuracy)


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()
print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3,
                      scoring='accuracy'))

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
print(confusion_matrix(y_train_5, y_train_pred))

print('presision score', precision_score(y_train_5, y_train_pred))
print('recall score', recall_score(y_train_5, y_train_pred))
print('f1 score', f1_score(y_train_5, y_train_pred))

y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method='decision_function')

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
