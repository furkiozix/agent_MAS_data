import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics

df = pd.read_csv('csvfile/abalone.csv')

df = df.replace('M', 0)
df = df.replace('F', 1)
df = df.replace('I', 2)


data = df.values
X = data[:,1:-1]
Y = data[:,0]


X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.33, random_state=42)

clf2 = tree.DecisionTreeClassifier()
clf2 = clf2.fit(X_train, y_train)


y_pred_en = clf2.predict(X_test)

#cofustion matrix
cm=confusion_matrix(y_test,y_pred_en)
print(cm)

#Accuracy value
result = clf2.score(X_test, y_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_en)*100)

#fpr and tpr

fpr, tpr, threshold = roc_curve(y_test, y_pred_en,pos_label=2)
roc_auc = auc(fpr, tpr)

#precision value
precision, recall, thresholds = precision_recall_curve(y_test,y_pred_en,pos_label=2)

#f1 score

#print(classification_report(y_test, y_pred_en))

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
plt.show()

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf2, X, Y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# K-cross validation test
from sklearn.model_selection import KFold

kf = KFold(n_splits=8)
print('10-fold cross validation method.....')

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test =  Y[train_idx], Y[test_idx]

    clf2.fit(X_train, y_train)
    score = clf2.score(X_test, y_test)
    print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
print(clf2.predict_proba(X)[:,1])
from sklearn.model_selection import cross_val_predict
t_pred = cross_val_predict(clf2, X, Y, cv=3, method='predict_proba')
print(t_pred[:,1])