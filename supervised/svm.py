import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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
    ('linear_svc', LinearSVC())
])

svm_clf.fit(X, y)
prediction = svm_clf.predict([[5.8, 1.7]])
print('prediction_', ['Not Iris-Virginica', 'Iris-Virginica'][prediction[0]])
