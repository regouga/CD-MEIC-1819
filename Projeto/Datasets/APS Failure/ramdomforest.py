#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 02:46:16 2018

@author: miguelregouga
"""

# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB



training = pd.read_csv("base_aps_failure_trainingCla.csv")
test = pd.read_csv("base_aps_failure_testCla.csv")
df_tr = pd.DataFrame(training)
df_te = pd.DataFrame(test)

from sklearn.ensemble import RandomForestClassifier




X_train = df_tr.iloc[:, 1:171] # Features
y_train = df_tr.iloc[:, 0]  # Labels

X_test = df_te.iloc[:, 1:171]
y_test = df_te.iloc[:, 0]



#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=2000)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



'''
df = pd.DataFrame(dataset)
print(df.head())
train, test = df[df['class']==True], df[df['class']==False]
features = df.columns[1:171]
print (features)


# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(train[features], df['class'])
'''