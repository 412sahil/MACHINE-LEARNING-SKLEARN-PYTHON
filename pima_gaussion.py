# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:04:55 2017

@author: Sahil Aggarwal
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:30:33 2017

@author: Sahil Aggarwal
"""

import csv
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


def loadData(filename):
    lines=csv.reader(open(filename,"rb"))
    pimaData = list(lines)
    pimaData.pop(0)
    for i in range(len(pimaData)):
        pimaData[i]=[float(x) for x in pimaData[i]]
    return pimaData
        
p = loadData('pimadiabetes.csv')
#pimaAttr = ['preg','Glucose','BP','skin','serum','bmi','PED','Age']
#pimaTarget = ['Not Diabetic','Diabetic']
pimaData = np.array(p)
a=pimaData[:,-1].astype(np.int64)

print(pimaData.shape)

y= pimaData[:,-1]
X= pimaData[:,0:8]
col1=normalize(X[:,0:8])
X[:,0:8]=col1
print "jugaad",X

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
estimator = GaussianNB()
estimator.fit(X_train, y_train)

print(estimator.score(X_test,y_test))

#estimator.fit(X_train,y_train)
y_pred=estimator.predict(X_test)

print ("acuraccy from GaussionNB after stan")
score =accuracy_score(y_test,y_pred)

print score

print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
print(cm)













        
        