# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from dummy_var import preprocessData
from sklearn.model_selection import *
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import *
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import plot_roc

bankcsv = pd.read_csv("/Users/miguelregouga/bank.csv")
bank = preprocessData(bankcsv)

X = bank.iloc[:, :]
print(X)

y = bankcsv['pep']

labels = pd.unique(y) 

print(str(labels))


trX, tsX, trY, tsY = train_test_split(X, y, train_size=0.7, stratify = y)


gnb = GaussianNB()

predY = gnb.fit(tsX, tsY).predict(tsX)

cm1 = confusion_matrix(tsY, predY, labels)

print(cm1)

print ("Pred: " + str(predY))





knn = KNeighborsClassifier(n_neighbors = 3)
model = knn.fit(trX, trY)
predY = model.predict(tsX)
cm2 = confusion_matrix(tsY, predY, labels)

clf = SVC(kernel='linear', C=10).fit(trX, trY)
scores = clf.score(tsX, tsY)
print("Score: " + str(scores))
