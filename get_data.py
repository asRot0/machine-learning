import pandas as pd
import os

dataset = './datasets/housing/housing.csv'


# pd.set_option('display.max_columns', None)
# pd.options.display.max_columns = None
os.environ['COLUMNS'] = '150'

data = pd.read_csv(dataset)
df = pd.DataFrame(data)
# gdp = pd.read_csv(data, thousands=',', delimiter='\t', na_values='n/a')

print(df.head())
