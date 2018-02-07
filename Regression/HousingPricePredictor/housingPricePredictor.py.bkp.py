import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits

data = pd.read_csv("housingPricePredictor.csv")
print(data.head())
print(data.describe())

'''
data['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine
plt.show()

plt.figure(figsize=(10,10))
sns.jointplot(x=data.lat.values, y=data.long.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()
sns.despine
'''

import math
subplotinrows = math.ceil(len(data.columns) / 3)
fig, axs = plt.subplots(subplotinrows, 3, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.5, wspace= 0.5)
axs = axs.ravel()
p = 0
print(list(data.columns.values), list(data.dtypes.values))
for col, dtype in zip(list(data.columns.values), list(data.dtypes.values)):
	#print(col, dtype)
	if dtype == "int64" or dtype == "float64":
		axs[p].scatter(data.price, data[col], s=1)
		axs[p].set_xlabel("Price")
		axs[p].set_ylabel(col)
	p += 1
plt.show()

labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)

from sklearn.cross_validation import train_test_split
x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)

from sklearn import model_selection
kfold = model_selection.KFold(n_splits=10, random_state=7)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
print("LinearRegression Score : ", reg.score(x_test,y_test))
reg.predict(x_test, y_test)

from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2, learning_rate = 0.1, loss = 'ls')
clf.fit(x_train, y_train)
print("GradientBoostingRegressor Score : ", clf.score(x_test,y_test))

