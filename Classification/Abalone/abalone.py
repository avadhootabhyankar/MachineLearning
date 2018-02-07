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

import pandas as pd
target_names = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
feature_names = ['Sex','Length','Diameter','Height','Whole Weight','Shucked Weight','Viscera Weight','Shell Weight']
data = pd.read_csv("abalone.data")
print("dataframe columns:",data.columns)
features = data.iloc[:,0:len(feature_names)]
print(features.head())
target = data.iloc[:,len(feature_names)]
print(target.head())
#print("Shape : ", len(features), len(features[0]))
print("Features : ", feature_names)
print("Classes : ", target_names)
print("X : \n", features)
print("y : \n", target)

features = labelEncode(features)
#print("After Label Encoding\n", features.head())
features = oneHotEncode(features, ['Sex']) #pass the list of columns that needs one hot encoding
#print("After One Hot Encoding\n", features.head())
features = standardize(features)
#print("After Standardization\n", features[0])

import math
def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

subplotinrowcol = math.ceil(math.sqrt(nCr(len(feature_names), 2)))
fig, axs = plt.subplots(subplotinrowcol, subplotinrowcol, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.5, wspace= 0.5)
axs = axs.ravel()
p = 0
for i in range(len(feature_names)):
	for j in range(i + 1, len(feature_names)):
		for m in target_names:
			axs[p].scatter(features[target==m,i],features[target==m,j], s=1)
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

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "Gaussian Process"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    GaussianProcessClassifier(1.0 * RBF(1.0))]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=.4, random_state=42)
for name, clf in zip(names, classifiers):
	clf.fit(X_train, y_train)

	s = pickle.dumps(clf)
	clf2 = pickle.loads(s)

	predictions = []
	for i in range(len(X_train)):
		prediction = clf2.predict(X_train[i:i+1])
		predictions.append(prediction[0])
		#print("Prediction : ", prediction[0], ", Actual : ", target[i])
	trainingAccuracy = accuracy_score(y_train, predictions)

	predictions = []
	for i in range(len(X_test)):
		prediction = clf2.predict(X_test[i:i+1])
		predictions.append(prediction[0])
		#print("Prediction : ", prediction[0], ", Actual : ", target[i])
	testingAccuracy = accuracy_score(y_test, predictions)

	print(name, " - Training Accuracy : ", trainingAccuracy, ", Testing Accuracy : ", testingAccuracy)