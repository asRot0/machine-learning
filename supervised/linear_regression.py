from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import random


def random_dot(x, y):
    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('random dots')
    plt.axis([0, 2, 0, 15])
    plt.legend()
    plt.show()


# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)

X = np.array([[2 * random.random()] for _ in range(100)])
y = np.array([[4 + 3 * x[0] + random.gauss(0, 1)] for x in X])


print('X', X[:5])
print('y', y[:5])
# random_dot(X, y)

x_b = np.c_[np.ones((100, 1)), X]
print('x_b', x_b[:5])

# theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
# print(theta_best)

x_new = np.array([[0], [2]])
# x_new_b = np.c_[np.ones((2, 1)), x_new]
# y_predict = np.dot(x_new_b, theta_best)

# plt.plot(x_new, y_predict, 'r-', label='prediction')
# plt.legend()
# random_dot(X, y)


lin_model = LinearRegression()
lin_model.fit(X, y)

print('intercept', lin_model.intercept_)
print('coef', lin_model.coef_)

y_predict = lin_model.predict(x_new)
print('y_predict', y_predict)

theta_best_svd, residuals, rank, s = np.linalg.lstsq(x_b, y, rcond=1e-6)
print('theta_best_svd', theta_best_svd)
print('residuals', residuals)
print('rank', rank)
print('s', s)