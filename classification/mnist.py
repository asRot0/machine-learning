from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

path = '../train models/mnist_sgd_clf.pkl'

mnist = fetch_openml('mnist_784', version=1)
print(mnist.keys())

X, y = mnist['data'], mnist['target'].astype(np.uint8)
# print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
some_digit = X.iloc[1]

'''
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
joblib.dump(sgd_clf, path)
'''

sgd_clf = joblib.load(path)
# print(sgd_clf.predict([some_digit]))

predict_label = sgd_clf.predict(X_train)
print('error_', np.sqrt(mean_squared_error(y_train, predict_label)))
print('error%_', np.mean(predict_label == y_train))
# print('error%new_', np.mean(sgd_clf.predict(X_test) == y_test))

'''
while True:
    i = int(input('iloc_'))
    print('original', y_train[i], end=' ')
    print('predict', sgd_clf.predict([X.iloc[i]]))
    print()
'''
