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
#sns.set_style('whitegrid')


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
k_neighbors = 100
GRAPHS_FOLDER = "randforest_graphs/"

def run_all_radfor(X, y, X_train, y_train, X_test, y_test):
    cross_val_vector = []
    precision = []
    recall = []
    accuracy =[]
    for n in range(1,k_neighbors,5):
        clf = RandomForestClassifier(n_estimators=n)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        print("\nRandomForest classifier with %d neighbors" % (n))
        #print(classification_report(y_test,y_pred,target_names=target_names))
        tn, fp, fn, tp = confusion_matrix(y_test,y_pred, labels=range(2)).ravel()
        print("TN: %d \tFP: %d \nFN: %d \tTP: %d" % (tn, fp, fn, tp))
        print("Accuracy score: %f" % (accuracy_score(y_test,y_pred)))
        print("ROC auc score: %f" % (roc_auc_score(y_test,y_pred)))
        clf = RandomForestClassifier(n_estimators=n)
        clf.fit(X,y)
        #print("Cross-Validation (10-fold) score: %f" % (cross_val_score(clf, X, y, cv=10).mean()))
        #cross_val_vector.append(cross_val_score(clf, X, y, cv=10).mean())
        precision.append(tp/(tp+fp))
        recall.append(tp/(tp+fn))
        accuracy.append((tp+tn)/(tp+fp+fn+tn))
    return (cross_val_vector, precision, recall, accuracy)



def draw_crossval_graph(res1, k_values, balance):
    #cross_val_vector = res1[0]
    #cross_val_vector_pca = res2[0]
    precision = res1[1]
    recall = res1[2]
    accuracy = res1[3]
    f = plt.figure()
    	
    plt.title("precision/recall score vs n-estimators with %s data" %(balance))
    plt.xlabel("n-estimators")
    plt.ylabel("precision/recall score")
    plt.gca().set_ylim([0.4,1])
    plt.grid()
    
    plt.plot(k_values, precision, '.-', color="r", label="precision")
    
    plt.plot(k_values, recall, '.-', color="g", label="recall")
    	
    plt.plot(k_values, accuracy, '.-', color="b", label="accuracy")
    
    plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))	
    
    f.savefig("%s%s_knn_%s.png" % (GRAPHS_FOLDER,"crossval",balance),bbox_inches="tight")
    f.savefig("%s%s_knn_%s.pdf" % (GRAPHS_FOLDER,"crossval",balance),bbox_inches="tight")
    
    


#training = pd.read_csv("base_aps_failure_trainingCla.csv")
#test = pd.read_csv("base_aps_failure_testCla.csv")
#df_tr = pd.DataFrame(training)
#df_te = pd.DataFrame(test)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0, stratify=y)


#X_train = df_tr.iloc[:, 1:171] # Features
#y_train = df_tr.iloc[:, 0]  # Labels

#X_test = df_te.iloc[:, 1:171]
#y_test = df_te.iloc[:, 0]


draw_crossval_graph(run_all_radfor(X, y, X_train, y_train, X_test, y_test),
                    list(range(1,k_neighbors,5)),
                    "unbalanced")

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