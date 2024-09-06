from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# os.environ['COLUMNS'] = '200'
dataset = 'datasets/housing/housing.csv'

housing = pd.read_csv(dataset)

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# housing['median_income'].hist()
# plt.show()
print(housing.info())

housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
print(housing.head())
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
print(housing.head())

# housing['income_cat'].hist()
# plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
print(split)

strat_train_set = pd.DataFrame
strat_test_set = pd.DataFrame

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(housing['income_cat'].value_counts() / len(housing))

for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)
# print(strat_test_set.head())

housing = pd.DataFrame(strat_train_set.copy())
'''
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population']/100, label='population', figsize=(10,7),
             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
plt.show()
'''

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

# housing.drop('ocean_proximity', axis=1, inplace=True)
corr_matrix = housing.corr(numeric_only=True)
# corr_matrix = housing.select_dtypes(include=[float, int]).corr()
print('correlations -----------------')
print(corr_matrix['median_house_value'].sort_values(ascending=False))

housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

'''
median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median, inplace=True)
print(median, type(median))
'''
# print(housing.drop('ocean_proximity', axis=1).median().values)

housing_num = housing.drop('ocean_proximity', axis=1)


'''
imputer = SimpleImputer(strategy='median')
imputer.fit(housing_num)
print(imputer.statistics_)

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

housing_cat = housing[['ocean_proximity']]
print(housing_cat.head(10))


"""
ordinal_encoder = OrdinalEncoder()
housing_cat_encoder = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoder[:10])
print(ordinal_encoder.categories_)
"""


cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(cat_encoder.categories_)
print(type(housing_cat_1hot))

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
# print(housing_extra_attribs)
'''


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
 ])
housing_num_tr = num_pipeline.fit_transform(housing_num)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
 ])
housing_prepared = full_pipeline.fit_transform(housing)

# ----------------- model
print('\n_ __'*10, '\n---> model\n')

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print('labels:', list(some_labels))
print('_ __ housing_labels.iloc[:5] ```````````````````````````')

# linear regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

print('[linear regression]~~~~~~\npredictions:', lin_reg.predict(some_data_prepared))

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print('error [linear regression]:', lin_rmse)

# decision tree regressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)

print('[decision tree regressor]~~~~~~\npredictions:', tree_reg.predict(some_data_prepared))
print('error [decision tree regressor]:', tree_rmse)

# using cross-validation for decision tree
'''
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
'''
'''
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)
print(tree_rmse_scores.mean())
'''

# using cross-validation for  linear regression
'''
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
'''

# random forest regressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)

print('[random forest regressor]~~~~~~\npredictions:', forest_reg.predict(some_data_prepared))
print('error [random forest regressor]:', forest_rmse)