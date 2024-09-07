from sklearn.datasets import load_digits
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import loadmat

mnist = load_digits()
print(mnist.keys())

X, y = mnist['data'], mnist['target']
print(X.shape)
print(y.shape)

some_digit = X[0]
some_digit_image = some_digit.reshape(8,8)

plt.imshow(some_digit_image, cmap=mpl.colormaps['cividis'], interpolation='nearest')
plt.axis('off')
plt.show()

data = loadmat('datasets/mnist/mnist-original.mat')
print(data.keys())
print(data.values())