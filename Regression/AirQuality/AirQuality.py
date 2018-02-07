from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

def labelEncode(features):
	from sklearn.preprocessing import LabelEncoder
	le=LabelEncoder()
	for col in features.columns.values:
		if features[col].dtypes=='object':
			data = features[col]
			le.fit(data.values)
			features[col] = le.transform(features[col])
	return features

def standardize(features):
	from sklearn.preprocessing import scale
	features=scale(features)
	return features

def oneHotEncode(features, columns):
	from sklearn.preprocessing import OneHotEncoder
	enc=OneHotEncoder(sparse=False)
	features_1=features
	for col in columns:
		data=features[[col]]
		enc.fit(data)
		temp = enc.transform(features[[col]])
		temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col].value_counts().index])
		temp=temp.set_index(features.index.values)
		features_1=pd.concat([features_1,temp],axis=1)
	return features

def minMaxScaler(features, columns):
	from sklearn.preprocessing import MinMaxScaler
	min_max=MinMaxScaler()
	features=min_max.fit_transform(features[columns])
	return features

def replaceMissingWithMean(data):
	import numpy
	print("Before : ", data.isnull().sum())
	data.fillna(data.mean(), inplace=True)
	print("After : ", data.isnull().sum())
	return data

import pandas as pd
feature_names = ['Time','CO(GT)','PT08.S1(CO)','NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)','PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','T','RH']
minMaxScalingColumns = ['CO(GT)','PT08.S1(CO)','NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)','PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','T','RH']
oneHotEncodeColumns = []
data = pd.read_csv("AirQuality.csv")
print("dataframe columns : ", data.columns)
print("before nan removal : ", data.shape)
#data.dropna(inplace=True)
#print("after nan removal : ", data.shape)
print("Before replacing NaN with Mean value : \n", data.head())
data = replaceMissingWithMean(data)
print("After replacing NaN with Mean value : \n", data.head())
features = data.iloc[0:500,1:len(feature_names)+1]
print("Features Head : \n", features.head())
target = data.iloc[0:500:,len(feature_names)+1]
target *= 10000
target = target.astype(int)
print("Target Head : \n", target.head())
#print("Shape : ", len(features), len(features[0]))
print("Features : ", feature_names)
#print("X : \n", features)
#print("y : \n", target)

features = labelEncode(features)
print("After Label Encoding\n", features.head())
features = oneHotEncode(features, oneHotEncodeColumns) #pass the list of columns that needs one hot encoding
print("After One Hot Encoding\n", features.head())
features = minMaxScaler(features, minMaxScalingColumns)
print("After Min Max Scaling\n", features[0])
features = standardize(features)
print("After Standardization\n", features[0])

import math
def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

subplotinrowcol = math.ceil(math.sqrt(nCr(len(feature_names), 2)))
fig, axs = plt.subplots(subplotinrowcol, subplotinrowcol, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.5, wspace= 0.5)
axs = axs.ravel()
p = 0
for i in range(len(feature_names)-1):
	for j in range(i + 1, len(feature_names)-1):
		print(i, j)
		axs[p].scatter(features[:,i],features[:,j], s=1)
		axs[p].set_xlabel(feature_names[i])
		axs[p].set_ylabel(feature_names[j])
		p+=1
plt.show()

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=.4, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
logreg = LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = logreg.predict(X_test)

print("Score:\n", logreg.score(X_test, y_test))

# The coefficients
print('Coefficients: \n', logreg.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

for yt, yp in zip(y_test, y_pred):
	print("Actual : ", yt, ", prediction : ", yp)

# Plot outputs
plt.scatter(X_test[:,0], y_test,  color='black')
plt.plot(X_test[:,0], y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()






