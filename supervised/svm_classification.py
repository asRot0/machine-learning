import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC


# Linear SVM Classification

iris = datasets.load_iris()
print(iris.keys())
print(iris.target_names, 'target_names')
print(iris.feature_names, 'feature_names')

print(iris.data[:5], 'iris.data[:5]')

X = iris['data'][:, (2, 3)]  # petal length, petal width
print(X[:5], 'X[:5]')

y = (iris.target == 2).astype(int)  # Iris-Virginica
print(y[:5], 'y[:5]')

svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC(C=1, loss='hinge'))
])

svm_clf.fit(X, y)
prediction = svm_clf.predict([[5.8, 1.7]])
print('prediction_', ['Not Iris-Virginica', 'Iris-Virginica'][prediction[0]])
print('__'*20)

# Nonlinear SVM Classification

moon = datasets.make_moons()
X_moon, y_moon = moon[0], moon[1]
print(X_moon[:5], y_moon[:5], sep='\n')

polynomial_svm_clf = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ('svm_clf', LinearSVC(C=10, loss='hinge'))
])

poly_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))
])

# polynomial_svm_clf.fit(X_moon, y_moon)
poly_kernel_svm_clf.fit(X_moon, y_moon)
print(X_moon[:5])