import pandas as pd
import matplotlib.pyplot as plt
import os

# pd.set_option('display.max_columns', None)
# pd.options.display.max_columns = None
os.environ['COLUMNS'] = '200'

dataset = 'datasets/housing'
savefig = 'plotfig'

housing = pd.read_csv(os.path.join(dataset, 'housing.csv'))

print(housing.head())
# print(housing.tail())
print(housing.info())
print(housing['ocean_proximity'].value_counts())
print(housing.describe())

housing.hist(bins=50, figsize=(10, 8))

# plt.pause(interval=2)
# plt.savefig(os.path.join(savefig, 'housing_data.png'))
plt.show()