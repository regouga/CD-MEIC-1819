# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 16:47:59 2018

@author: João Pina
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss


training_dataset = pd.read_csv("aps_failure_training_set.csv", sep=',', header=14, engine='python')

print("CLUSTERS")
training_dataset['class'] = training_dataset['class'].replace('neg', 0)
training_dataset['class'] = training_dataset['class'].replace('pos', 1)


print("CRUNCHY")
training_dataset.head()
print("CHOCO")
training_dataset.shape
print("CHOCOLATE")
training_dataset.columns
print("NESTLE")

training_dataset.isnull().sum()
print("NOVO")
training_dataset.dropna(inplace = True)
print("CONA")
#training_dataset = pd.get_dummies(training_dataset, drop_first = True)
print("FALE")

training_dataset['class'].value_counts()
print("CONNOSCO")

X = training_dataset.drop("class", axis = 1)
y = training_dataset['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, stratify=y)
y_train.value_counts()
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)

recall_score(y_test, y_pred)





X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, stratify=y)

smt = SMOTE()
X_train, y_train = smt.fit_sample(X_train, y_train)
np.bincount(y_train)

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)

recall_score(y_test, y_pred)




#training_dataset.to_csv("base_aps_failure_training.csv", index=False)


test_dataset = pd.read_csv("aps_failure_test_set.csv", sep=',', header=14, engine='python')

test_dataset['class'] = test_dataset['class'].replace('neg', 0)
test_dataset['class'] = test_dataset['class'].replace('pos', 1)




#test_dataset.to_csv("base_aps_failure_test.csv", index=False)



