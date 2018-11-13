# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 19:23:03 2018

@author: Hélio
"""
from sklearn.decomposition import PCA
#from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("base_aps_failure_trainingCla.csv", sep=',', engine='python')
X = np.array(data.drop("class", axis=1))
y = np.array(data["class"])
target_names = np.array(["neg","pos"])
print("eeieie")
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0, stratify=y)

# Over-sampling techniques
sm = SMOTE(random_state=1)
X_sm, y_sm = sm.fit_sample(X,y)
X_train_sm, y_train_sm = sm.fit_sample(X_train,y_train)
