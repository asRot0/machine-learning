import pandas as pd
import matplotlib.pyplot as plt
import os

# pd.set_option('display.max_columns', None)
# pd.options.display.max_columns = None
os.environ['COLUMNS'] = '200'

dataset = './datasets/housing/housing.csv'
housing = pd.read_csv(dataset)

print(housing.head())
# print(housing.tail())
print(housing.info())
print(housing['ocean_proximity'].value_counts())
print(housing.describe())

housing.hist(bins=50, figsize=(20, 15))
plt.show()