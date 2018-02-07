import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import skew
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import norm
from collections import Counter
from sklearn.linear_model import LinearRegression,LassoCV, Ridge, LassoLarsCV,ElasticNetCV
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
warnings.filterwarnings('ignore')
sns.set(style='white', context='notebook', palette='deep')

# Load train and Test set
train = pd.read_csv("adult.train")
test = pd.read_csv("adult.test")

print("Train data size before dropping Id feature is : {}".format(train.shape))
print("Test data size before dropping Id feature is : {}".format(test.shape))

train_ID = train['id']
test_ID = test['id']

# Now drop the 'Id' column since it's unnecessary for the prediction process.
train.drop('id', axis = 1, inplace = True)
test.drop('id', axis = 1, inplace = True)

print("Train data size before dropping Id feature is : {}".format(train.shape))
print("Test data size before dropping Id feature is : {}".format(test.shape))

print(train.head())
print(test.head())