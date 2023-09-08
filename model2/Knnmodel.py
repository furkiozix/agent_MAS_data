from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pandas import read_csv

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

filename = 'csvfile/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

array = dataframe.values              # Convert dataframe to numpy array 
#Model Making
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33                      # Split test 33.33%, training 63.33%

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state = 7)

model =  KNeighborsClassifier(n_neighbors = 15, metric='euclidean')
model.fit(X_train, Y_train)

#model Evaluation

#Accuracy value
result = model.score(X_test, Y_test)
#print("Accuracy %.2f%%" % (result*100.0))

#fpr and tpr
Y_scores = model.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(Y_test, Y_scores[:, 1])
roc_auc = auc(fpr, tpr)

#precision value
precision, recall, thresholds = precision_recall_curve(Y_test,Y_scores[:, 1])

yhat = model.predict(X_test)
#f1 score
#print(classification_report(Y_test,yhat))

#cofustion matrix

cm=confusion_matrix(Y_test,yhat)
#print(cm)
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
scores = cross_val_score(model, X, Y, cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# K-cross validation test
from sklearn.model_selection import KFold

kf = KFold(n_splits=8)
#print('10-fold cross validation method.....')

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test =  Y[train_idx], Y[test_idx]

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
#print(model.predict_proba(X)[:,1])
from sklearn.model_selection import cross_val_predict
t_pred = cross_val_predict(model, X, Y, cv=3, method='predict_proba')
#print(t_pred[:,1])