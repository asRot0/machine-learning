from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, \
    roc_curve, roc_auc_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

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


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b-', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')

    # Set the threshold to 0.5
    threshold = 0.5

    # Find the closest threshold to 0.5
    closest_index = np.argmin(np.abs(thresholds - threshold))

    # Highlight the threshold with a vertical line
    plt.axvline(x=thresholds[closest_index], color='r', linestyle='--',
                label=f'Threshold = {thresholds[closest_index]:.2f}')

    # Get precision and recall at the closest threshold
    precision_at_threshold = precisions[closest_index]
    recall_at_threshold = recalls[closest_index]

    # Highlight precision and recall points
    plt.plot(thresholds[closest_index], precision_at_threshold, 'bo')  # Blue dot for precision
    plt.plot(thresholds[closest_index], recall_at_threshold, 'go')  # Green dot for recall

    # Vertical line at the threshold (already drawn with axvline above)

    # Horizontal lines at precision and recall points
    plt.axhline(y=precision_at_threshold, color='b', linestyle='--', label=f'Precision = {precision_at_threshold:.2f}')
    plt.axhline(y=recall_at_threshold, color='g', linestyle='--', label=f'Recall = {recall_at_threshold:.2f}')

    # Labels, grid, and legend
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.legend(loc='best')
    plt.grid(True)
    # plt.savefig(os.path.join('../plotfig', 'mnist_precision_recall_vs_threshold.png'))


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Training a Binary Classifier

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

print('precision score', precision_score(y_train_5, y_train_pred))
print('recall score', recall_score(y_train_5, y_train_pred))
print('f1 score', f1_score(y_train_5, y_train_pred))

y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method='decision_function')

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
print('threshold_90_precision', threshold_90_precision)

y_train_pred_90 = (y_scores >= threshold_90_precision)
print('precision score', precision_score(y_train_5, y_train_pred_90))
print('recall score', recall_score(y_train_5, y_train_pred_90))

print('roc_auc score(sgd_clf)', roc_auc_score(y_train_5, y_scores))
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method='predict_proba')
y_scores_forest = y_probas_forest[:, 1]  # proba of positive class

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
plt.plot(fpr, tpr, 'b:', label='SGD')
plot_roc_curve(fpr_forest, tpr_forest, 'Random Forest')
plt.legend(loc='lower right')
plt.show()
print('roc_auc_score(forest_clf)', roc_auc_score(y_train_5, y_scores_forest))
