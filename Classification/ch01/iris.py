from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

def labelEncode(X_train, X_test):
	#Label encoding
	from sklearn.preprocessing import LabelEncoder
	le=LabelEncoder()
	# Iterating over all the common columns in train and test
	for col in X_test.columns.values:
		# Encoding only categorical variables
		if X_test[col].dtypes=='object':
			# Using whole data to form an exhaustive list of levels
			data=X_train[col].append(X_test[col])
			le.fit(data.values)
			X_train[col]=le.transform(X_train[col])
			X_test[col]=le.transform(X_test[col])
	return X_train, X_test

def standardize(X_train, X_test):
	from sklearn.preprocessing import scale
	X_train=scale(X_train)
	X_test=scale(X_test)
	return X_train, X_test

def oneHotEncode(X_train, X_test, columns):
	from sklearn.preprocessing import OneHotEncoder
	enc=OneHotEncoder(sparse=False)
	X_train_1=X_train
	X_test_1=X_test
	for col in columns:
		# creating an exhaustive list of all possible categorical values
		data=X_train[[col]].append(X_test[[col]])
		enc.fit(data)
		# Fitting One Hot Encoding on train data
		temp = enc.transform(X_train[[col]])
		# Changing the encoded features into a data frame with new column names
		temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col].value_counts().index])
		# In side by side concatenation index values should be same
		# Setting the index values similar to the X_train data frame
		temp=temp.set_index(X_train.index.values)
		# adding the new One Hot Encoded varibales to the train data frame
		X_train_1=pd.concat([X_train_1,temp],axis=1)
		# fitting One Hot Encoding on test data
		temp = enc.transform(X_test[[col]])
		# changing it into data frame and adding column names
		temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col].value_counts().index])
		# Setting the index for proper concatenation
		temp=temp.set_index(X_test.index.values)
		# adding the new One Hot Encoded varibales to test data frame
		X_test_1=pd.concat([X_test_1,temp],axis=1)
	return X_train, X_test

f = open("data/iris.csv")
shape = f.readline().replace("\n","").split(',')
target_names = f.readline().replace("\n","").split(',')
feature_names = f.readline().replace("\n","").split(',')
data = np.loadtxt(f, delimiter=',')
features = data[:,0:len(data[0])-1]
target = data[:,len(data[0])-1].astype(int)
print("Shape : ", shape[0], shape[1])
print("Features : ", feature_names)
print("Classes : ", target_names)
print("X : \n", features)
print("y : \n", target)

'''
data = load_iris()
#print(data)
features = data['data']
feature_names = data['feature_names']
target = data['target']
target_names = data['target_names']
'''

plt.figure(figsize=(4, 8))
plt.subplots_adjust(bottom=0, top=1, left=.05, right=.95, wspace=0.5, hspace=0.5)
k = 521
for i in range(len(feature_names)):
	for j in range(i + 1, len(feature_names)):
		plt.subplot(k)
		plt.xlabel(feature_names[i])
		plt.ylabel(feature_names[j])
		for t,marker,c in zip(range(len(target_names)),">ox","rgb"): #Markers and Colors are optional on below line e.g. marker=marker, c=c
			plt.scatter(features[target == t,i], features[target == t,j])
		k += 1
#plt.show()

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

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=.4, random_state=42)
print("Before preprocessing :\n", X_train[0])
import pandas as pd
X_train = pd.DataFrame(data = X_train)
X_test = pd.DataFrame(data = X_test)
X_train, X_test = labelEncode(X_train, X_test)
print("After Label Encoding\n", X_train.head())
X_train, X_test = oneHotEncode(X_train, X_test, []) #pass the list of columns that needs one hot encoding
print("After One Hot Encoding\n", X_train.head(), "X_train size", len(X_train))
X_train, X_test = standardize(X_train, X_test)
print("After Standardization\n", X_train[0], "X_train size", len(X_train))

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


import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt

digits = load_iris()
X, y = digits.data, digits.target

title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

plt.show()













