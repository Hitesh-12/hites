# -*- coding: utf-8 -*-
# Data Preprocessing

# Importing the libraries
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)
#import scipy.stats as stats
#import statsmodels.api as sm

# Importing the dataset
train = pd.read_csv('C:/Users/Hitesh Sharma/Desktop/facebook.csv')

X = train.iloc[:, :-1] #Take all the columns except last one
y = train.iloc[:, -1] #Take the last column as the result

print(train)

#EDA
sns.countplot(x='Country',data=train,palette='RdBu_r')
sns.distplot(train['Time Spent on Site'].dropna(),kde=False,color='darkred',bins=30)
sns.countplot(x='Salary',data=train)
sns.boxplot(x='Country',y='Salary',data=train,palette='winter')

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Taking care of missing data


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 3] = labelencoder_X.fit_transform(X.iloc[:, 3])

#Make dummy variables
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train.iloc[:, [0,5]] = sc.fit_transform(X_train.iloc[:, [0,5]])
X_test.iloc[:, [0,5]] = sc.transform(X_test.iloc[:, [0,5]])


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#Find relevant features
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X_train, y_train)
print("Optimal number of features : %d" % rfecv.n_features_)


# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


from sklearn.feature_selection import RFE

rfe = RFE(classifier, rfecv.n_features_, step=1)
rfe = rfe.fit(X_train, y_train.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
y_pred = classifier.predict(X_test)


# K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
model_accuracy = accuracies.mean()
model_standard_deviation = accuracies.std()

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


#ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
area_under_curve = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % area_under_curve)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()






#sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


#cr=df.corr()

# Encoding categorical data
# Encoding the Independent Variable
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X.iloc[:, 3] = labelencoder_X.fit_transform(X.iloc[:, 3])

#Make dummy variables
#onehotencoder = OneHotEncoder(categorical_features = [3])
#X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
#X = X[:, 1:]

