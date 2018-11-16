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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("col_classification.csv")
X = np.array(data.drop("consensus",axis=1).
             drop("experts::0",axis=1).
             drop("experts::1",axis=1).
             drop("experts::2",axis=1).
             drop("experts::3",axis=1).
             drop("experts::4",axis=1).
             drop("experts::5",axis=1))
y = np.array(data["consensus"])
GRAPHS_FOLDER = "randforest_graphs/"

def run_all_radfor(X, y, X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_estimators=70)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test,y_pred)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0, stratify=y)



yaxis = [run_all_radfor(X, np.array(data["consensus"]), X_train, y_train, X_test, y_test),
     run_all_radfor(X, np.array(data["experts::0"]), X_train, y_train, X_test, y_test),
     run_all_radfor(X, np.array(data["experts::1"]), X_train, y_train, X_test, y_test),
     run_all_radfor(X, np.array(data["experts::2"]), X_train, y_train, X_test, y_test),
     run_all_radfor(X, np.array(data["experts::3"]), X_train, y_train, X_test, y_test),
     run_all_radfor(X, np.array(data["experts::4"]), X_train, y_train, X_test, y_test),
     run_all_radfor(X, np.array(data["experts::5"]), X_train, y_train, X_test, y_test)]

x = ["consensus", "experts::0", "experts::1", "experts::2", "experts::3", "experts::4", "experts::5"]
width = 1/1.5
plt.bar(x, yaxis, width, color="blue")
plt.title("accuracy")
plt.gca().set_ylim([0.7,0.9])

fig = plt.gcf()
plotly_fig = tls.mpl_to_plotly(fig)
py.iplot(plotly_fig, filename='mpl-basic-bar')


#Create a Gaussian Classifier
#clf=RandomForestClassifier(n_estimators=1000)

#Train the model using the training sets y_pred=clf.predict(X_test)
#clf.fit(X_train,y_train)

#y_pred=clf.predict(X_test)

#draw_crossval_graph()

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


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