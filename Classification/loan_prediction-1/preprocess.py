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

# Importing pandas
import pandas as pd
# Importing training data set
X_train=pd.read_csv('X_train.csv')
Y_train=pd.read_csv('Y_train.csv')
# Importing testing data set
X_test=pd.read_csv('X_test.csv')
Y_test=pd.read_csv('Y_test.csv')
print("X_train head : \n",X_train.head())

print("Before Label Encoding:\n", X_train.head())
X_train, X_test = labelEncode(X_train, X_test)
print("After Label Encoding:\n", X_train.head())

columns=['Gender', 'Married', 'Dependents', 'Education','Self_Employed',
	          'Credit_History', 'Property_Area']
X_train, X_test = oneHotEncode(X_train, X_test, columns)
print("After One Hot Encoding:\n", X_train.head())

#X_train, X_test = standardize(X_train, X_test)
#print("After Standardization:\n", X_train.head())
from sklearn.preprocessing import scale
X_train=scale(X_train)
X_test=scale(X_test)

from sklearn.linear_model import LogisticRegression
log=LogisticRegression(penalty='l2',C=1)
log.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score
# Checking the model's accuracy
print(accuracy_score(Y_test,log.predict(X_test)))