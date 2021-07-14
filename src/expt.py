import dvc.api
import pandas as pd

with dvc.api.open(repo="https://github.com/raja-7-c/MLOps_Assignment", path="data/creditcard.csv", mode="r") as fd:
    df = pandas.read_csv(fd)

import numpy as np

#creating target series
target=df['Class']
#print(target)

#dropping the target variable from the data set
df.drop('Class',axis=1,inplace=True)
print(df.shape)

#converting them to numpy arrays
X=np.array(df)
y=np.array(target)
print(X.shape)
print(y.shape)

#distribution of the target variable
print(len(y[y==1]))
print(len(y[y==0]))

import numpy as np
# Load libraries
import pandas as pd
#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

#splitting the data set into train and test (80:20)
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training and 20% test
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


df1 = pd.DataFrame(X_train)
df1.to_csv("data/prepared/train.csv")

df2 = pd.DataFrame(X_test)
df2.to_csv("data/prepared/test.csv")

#applyting SMOTE to oversample the minority class
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=2)
X_sm,y_sm=sm.fit_resample(X_train,y_train)
print(X_sm.shape,y_sm.shape)
print(len(y_sm[y_sm==1]),len(y_sm[y_sm==0]))

#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(random_state=3)
rf.fit(X_sm,y_sm)
y_rf=rf.predict("data/prepared/test.csv")
y_rf_prob=rf.predict_proba("data/prepared/test.csv")[:,1]


import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score
print("F1_Score: \n",f1_score(y_test,y_rf,average = 'weighted'))
#Performance metrics evaluation
print("Confusion Matrix:\n",metrics.confusion_matrix(y_test,y_rf))
print("Accuracy:\n",metrics.accuracy_score(y_test,y_rf))
print("Precision:\n",metrics.precision_score(y_test,y_rf))
print("Recall:\n",metrics.recall_score(y_test,y_rf))
print("AUC:\n",metrics.roc_auc_score(y_test,y_rf_prob))
auc=metrics.roc_auc_score(y_test,y_rf_prob)

import json
f1score = f1_score(y_test,y_rf,average = 'weighted')
acc = metrics.accuracy_score(y_test,y_rf)
acc_f1_dict = {
    'F1_Score' : f1score,
    'Accuracy':
 acc }

out_file = open("metrics/acc_f1.json","w")
json.dump(acc_f1_dict,out_file)
out_file.close()

import matplotlib.pyplot as plt
#plotting the ROC curve
fpr,tpr,thresholds=metrics.roc_curve(y_test,y_rf_prob)
plt.plot(fpr,tpr,'b', label='AUC = %0.2f'% auc)
plt.plot([0,1],[0,1],'r-.')
plt.xlim([-0.2,1.2])
plt.ylim([-0.2,1.2])
plt.title('Receiver Operating Characteristic\nRandom Forest')
plt.legend(loc='lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

feature_cols = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11",
                "V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]

import pickle
random_forest_model_filename = 'models/model.pkl'
random_forest_model_pkl = open(random_forest_model_filename,'wb')
pickle.dump(rf,random_forest_model_pkl)
random_forest_model_pkl.close()

