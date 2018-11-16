# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 22:11:32 2018

@author: Hélio
"""


import pandas as pd
import numpy as np
#import graphviz 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectPercentile
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

data = pd.read_csv("col_classification.csv")
X = np.array(data.drop("consensus",axis=1).
             drop("experts::0",axis=1).
             drop("experts::1",axis=1).
             drop("experts::2",axis=1).
             drop("experts::3",axis=1).
             drop("experts::4",axis=1).
             drop("experts::5",axis=1))
y = np.array(data["consensus"])
target_names = np.array(['0','1'])
feature_names = list(data)[1:]
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)


importances = clf.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
model = SelectFromModel(clf, prefit=True)
X = model.transform(X)
print("shape do x; ", X.shape)   

new_features = []
for i in indices[:X.shape[1]]:
    new_features += [feature_names[i]]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0, stratify=y)


def decisionTree(X_train, y_train, X_test, y_test, min_sample_leaf, min_samples_split):	
    clf = DecisionTreeClassifier(min_samples_leaf =min_sample_leaf, min_samples_split=min_samples_split, random_state=0)    
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("accuracy:", accuracy_score(y_test,y_pred))
    return accuracy_score(y_test,y_pred)

accuracy = 0
i_max = 0
j_max = 0
for i in range(1,4000,100):
    for j in range(2, 4000, 100):
        accuracy_new = decisionTree(X, y, X_test, y_test, i,j)
        print(i, j, accuracy_new)
        if (accuracy < accuracy_new):
            accuracy = accuracy_new
            i_max = i
            j_max = j
print(i_max, j_max,accuracy)
