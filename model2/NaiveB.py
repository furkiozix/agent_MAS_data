import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# loading the iris dataset

data=pd.read_csv('csvfile/iris.csv')
data.variety.replace(['Setosa', 'Versicolor', 'Virginica'], [1, 2, 3], inplace=True)
array = data.values
# X -> features, y -> label
X = array[:,0:4]
y = array[:,4]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)


# dividing X, y into train and test data


# training a Naive Bayes classifier

gnb = GaussianNB()
y_pred = gnb.fit(X, y).predict(X)

#cofustion matrix
cm=confusion_matrix(y,y_pred)
#print(cm)

#Accuracy value
result = gnb.score(X_test, y_test)
#print("Accuracy %.2f%%" % (result*100.0))

#fpr and tpr

fpr, tpr, threshold = roc_curve(y, y_pred,pos_label=3)
roc_auc = auc(fpr, tpr)

#precision value
precision, recall, thresholds = precision_recall_curve(y,y_pred,pos_label=3)

#f1 score

#print(classification_report(y, y_pred))
"""
#plot
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.plot(recall, precision,'orange',label ='precision-recall curve for the model')
plt.legend(loc = 'lower right')
plt.title('ROC Curve of kNN')
plt.show()"""

from sklearn.model_selection import cross_val_score
scores = cross_val_score(gnb, X, y, cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# K-cross validation test
from sklearn.model_selection import KFold

kf = KFold(n_splits=8)
#print('10-fold cross validation method.....')

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test =  y[train_idx], y[test_idx]

    gnb.fit(X_train, y_train)
    score = gnb.score(X_test, y_test)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
#print(gnb.predict_proba(X)[:,1])
from sklearn.model_selection import cross_val_predict
t_pred = cross_val_predict(gnb, X, y, cv=3, method='predict_proba')
#print(t_pred[:,1])