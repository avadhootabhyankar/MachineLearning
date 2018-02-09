import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
import matplotlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "."]).decode("utf8"))

# Any results you write to the current directory are saved as output.

data = pd.read_csv('IsUpselledMini.csv')
print(data.head())

col = data.columns
print(col)

# y includes our labels and x includes our features
y = data.IsUpselled
trainId = data.EasyId
#EasyId,EntryStatus,Gender,Birthdate,A1,A2,A2-1,A2-2,IsUpselled,ApplyYear,ApplyMonth
list = ['EasyId','IsUpselled','EntryStatus']
x = data.drop(list,axis = 1 )
#x.head()
print(x.describe())
ax = sns.countplot(x.A1, label="カウント")
plt.show()

ax = sns.countplot(x.A2, label="カウント")
plt.show()

#ax = sns.countplot(x[['A2-1']], label="カウント")
#plt.show()

#ax = sns.countplot(x[['A2-2']], label="カウント")
#plt.show()

ax = sns.countplot(x.ApplyYear, label="カウント")
plt.show()

ax = sns.countplot(x.ApplyMonth, label="カウント")
plt.show()

'''

# Find Missing Ratio of Dataset
print("Missing values")
x_na = (x.isnull().sum() / len(x)) * 100
x_na = x_na.drop(x_na[x_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :x_na})
print(missing_data)

from sklearn.preprocessing import LabelEncoder
cols = ('Gender','A1','A2','A2-1','A2-2','ApplyYear','ApplyMonth')
# Process columns and apply LabelEncoder to categorical features
for c in cols:
	print(c)
	lbl = LabelEncoder() 
	lbl.fit(x[c].values.tolist()) 
	x[c] = lbl.transform(x[c].values.tolist())

ax = sns.countplot(y, label="Count")       # M = 212, B = 357
N, Y = y.value_counts()
print('Number of Upselled: ', Y)
print('Number of NonUpselled : ',N)
print('Upsell Ration : ', Y/N*100)
plt.show()

sns.set(style="whitegrid", palette="muted")
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[0:100,0:7]],axis=1)
data = pd.melt(data,id_vars="IsUpselled",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="IsUpselled", data=data)
plt.xticks(rotation=90)
plt.show()

##########################################################################
# 1. Feature selection with correlation and random forest classification #
##########################################################################

x_1 = x
#print(x_1.head())
print("Gender : ", x_1.Gender.unique())
print("A1 : ", x_1.A1.unique())
print("A2 : ", x_1.A2.unique())
#print("A2-1 : ", x_1.A2-1.unique())
#print("A2-2 : ", x_1.A2-2.unique())
print("ApplyYear : ", x_1.ApplyYear.unique())
print("ApplyMonth : ", x_1.ApplyMonth.unique())

#correlation map
f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(x_1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3, random_state=42)

#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier()      
clr_rf = clf_rf.fit(x_train,y_train)

ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")
plt.show()

proba = clf_rf.predict_proba(x_test)[:,1]
proba.sort()
id = pd.Series(range(1, len(proba) + 1))
plt.figure(figsize=(4, 8))
plt.xlabel("Test #")
plt.ylabel("Probablity")
plt.scatter(id, proba)
plt.show()

####################################################################
# 2) Univariate feature selection and random forest classification #
####################################################################

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
# find best scored 5 features
select_feature = SelectKBest(chi2, k=5).fit(x_train, y_train)
print('Score list:', select_feature.scores_)
print('Feature list:', x_train.columns)

selector = SelectKBest(chi2, k=5)
selector.fit(x_train, y_train)
x_new = selector.transform(x_train)
print('Chosen best 5 feature by SelectKBest:', x_train.columns[selector.get_support(indices=True)].tolist())

x_train_2 = select_feature.transform(x_train)
x_test_2 = select_feature.transform(x_test)
#random forest classifier with n_estimators=10 (default)
clf_rf_2 = RandomForestClassifier(n_estimators=500)      
clr_rf_2 = clf_rf_2.fit(x_train_2,y_train)
ac_2 = accuracy_score(y_test,clf_rf_2.predict(x_test_2))
print('Accuracy is: ',ac_2)
cm_2 = confusion_matrix(y_test,clf_rf_2.predict(x_test_2))
sns.heatmap(cm_2,annot=True,fmt="d")
plt.show()

proba = clr_rf_2.predict_proba(x_test_2)[:,1]
proba.sort()
id = pd.Series(range(1, len(proba) + 1))
plt.figure(figsize=(4, 8))
plt.xlabel("Test #")
plt.ylabel("Probablity")
plt.scatter(id, proba)
plt.show()

#############################################################
# 3) Recursive feature elimination (RFE) with random forest #
#############################################################

from sklearn.feature_selection import RFE
# Create the RFE object and rank each pixel
clf_rf_3 = RandomForestClassifier()      
rfe = RFE(estimator=clf_rf_3, n_features_to_select=5, step=1)
rfe = rfe.fit(x_train, y_train)
print('Chosen best 5 feature by rfe:', x_train.columns[rfe.support_])

x_train_3 = rfe.transform(x_train)
x_test_3 = rfe.transform(x_test)

clf_rf_3 = RandomForestClassifier()      
clr_rf_3 = clf_rf_3.fit(x_train_3,y_train)
print(clr_rf_3.predict_proba(x_test_3))

ac_3 = accuracy_score(y_test,clf_rf_3.predict(x_test_3))
print('Accuracy is: ',ac_3)
cm_3 = confusion_matrix(y_test,clf_rf_3.predict(x_test_3))
sns.heatmap(cm_3,annot=True,fmt="d")
plt.show()

###########################################################################################
# 4) Recursive feature elimination with cross validation and random forest classification #
###########################################################################################

from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestClassifier() 
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])

# Plot number of features VS. cross-validation scores
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

####################################################################
# 5) Tree based feature selection and random forest classification #
####################################################################

clf_rf_5 = RandomForestClassifier()      
clr_rf_5 = clf_rf_5.fit(x_train,y_train)
importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()

######################
# Feature Extraction #
######################

# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#normalization
x_train_N = (x_train-x_train.mean())/(x_train.max()-x_train.min())
x_test_N = (x_test-x_test.mean())/(x_test.max()-x_test.min())

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(x_train_N)

plt.figure(1, figsize=(14, 13))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_ratio_')
plt.show()
'''