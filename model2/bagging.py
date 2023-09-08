
# Bagged Decision Trees for Classification
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

data = pd.read_csv("csvfile/bank.csv")

le = LabelEncoder()
for col in data.columns:
    if data[col].dtypes =='object':            ####### when column's data type is equal to object
        data[col] = le.fit_transform(data[col])     ###### fit_transform is used for conversion



array = data.values
X = array[:,0:16]
Y = array[:,16]
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2)

LDA = LinearDiscriminantAnalysis( n_components=1)

X_train = LDA.fit_transform(X_train, y_train)
X_test = LDA.transform(X_test)

seed = 3
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed).fit(X_train,y_train)

#accuracy
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
#print(results.mean())

y_pred_en = model.predict(X_test)
#cofustion matrix
cm=confusion_matrix(y_test,y_pred_en)
#print(cm)

#fpr and tpr

fpr, tpr, threshold = roc_curve(y_test, y_pred_en)
roc_auc = auc(fpr, tpr)

#precision value
precision, recall, thresholds = precision_recall_curve(y_test,y_pred_en)

#f1 score

#print(classification_report(y_test, y_pred_en))
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

scores = cross_val_score(model, X, Y, cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# K-cross validation test
from sklearn.model_selection import KFold

kf = KFold(n_splits=10)
#print('10-fold cross validation method.....')

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test =  Y[train_idx], Y[test_idx]

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
from sklearn.model_selection import cross_val_predict
t_pred = cross_val_predict(model, X, Y, cv=3, method='predict_proba')
#print(t_pred[:,1])