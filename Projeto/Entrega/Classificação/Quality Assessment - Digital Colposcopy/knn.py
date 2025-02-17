# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 17:39:45 2018

@author: Hélio
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

GRAPHS_FOLDER = "knn_graphs/"
data = pd.read_csv("col_classification.csv")
X = np.array(data.drop("consensus",axis=1).
             drop("experts::0",axis=1).
             drop("experts::1",axis=1).
             drop("experts::2",axis=1).
             drop("experts::3",axis=1).
             drop("experts::4",axis=1).
             drop("experts::5",axis=1))
e = ["consensus","experts::0", "experts::1", "experts::2", "experts::3", "experts::4","experts::5"]
expert = ["consensus","experts0", "experts1", "experts2", "experts3", "experts4","experts5"]

for i in range(7):
    data = pd.read_csv("col_classification.csv")
    X = np.array(data.drop("consensus",axis=1).
             drop("experts::0",axis=1).
             drop("experts::1",axis=1).
             drop("experts::2",axis=1).
             drop("experts::3",axis=1).
             drop("experts::4",axis=1).
             drop("experts::5",axis=1))
    e = ["consensus","experts::0", "experts::1", "experts::2", "experts::3", "experts::4","experts::5"]
    expert = ["consensus","experts0", "experts1", "experts2", "experts3", "experts4","experts5"]
    y = np.array(data[e[i]])
    sm = SMOTE(random_state=1)
    X, y = sm.fit_sample(X, y)
    labels = pd.unique(y)
    k_neighbors = 100
    
    def run_all_knn(X, y, X_train, y_train, X_test, y_test):
        cross_val_vector = []
        precision = []
        recall = []
        accuracy = []
        for n in range(1,k_neighbors,4):
            clf = KNeighborsClassifier(n_neighbors=n)
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            print("\nKNN classifier with %d neighbors" % (n))
            #print(classification_report(y_test,y_pred,target_names=target_names))
            tn, fp, fn, tp = confusion_matrix(y_test,y_pred, labels=range(2)).ravel()
            print("TN: %d \tFP: %d \nFN: %d \tTP: %d" % (tn, fp, fn, tp))
            print("Accuracy score: %f" % (accuracy_score(y_test,y_pred)))
            print("ROC auc score: %f" % (roc_auc_score(y_test,y_pred)))
            clf = KNeighborsClassifier(n_neighbors=n)
            clf.fit(X,y)
            #print("Cross-Validation (10-fold) score: %f" % (cross_val_score(clf, X, y, cv=10).mean()))
            cross_val_vector.append(cross_val_score(clf, X, y, cv=10).mean())
            precision.append(precision_score(y_test,y_pred))
            recall.append(recall_score(y_test,y_pred))
            accuracy.append(accuracy_score(y_test,y_pred))
        return (cross_val_vector, precision, recall, accuracy)
    
    def draw_crossval_graph(res1, res2, k_values, balance):
        cross_val = res1[0]
        #cross_val_vector_pca = res2[0]
        precision = res1[1]
        #precision_pca = res2[1]
        recall = res1[2]
        #recall_pca = res2[2]
        accuracy = res1[3]
        f = plt.figure()
        	
        plt.title("metrics score vs k-neighbors")
        plt.xlabel("k-neighbors")
        plt.ylabel("metrics score")
        if (balance == 'unbalanced'):
            plt.gca().set_ylim([0.4,1])
        else:
            plt.gca().set_ylim([0.6,1])
        plt.grid()
        
        plt.plot(k_values, precision, '.-', color="r", label="precision")
        plt.plot(k_values, cross_val, '.-', color="b", label="cross_val")
        
        plt.plot(k_values, recall, '.-', color="g", label="recall")
        #plt.plot(k_values, recall_pca, '.-', color="y", label="recall with PCA")
        plt.plot(k_values, accuracy, '.-', color="y", label="accuracy")
        plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))	
        
        f.savefig("%s%s_knn_%s.png" % (GRAPHS_FOLDER,expert[i],balance),bbox_inches="tight")
        f.savefig("%s%s_knn_%s.pdf" % (GRAPHS_FOLDER,expert[i],balance),bbox_inches="tight")
    
    
    # split dataset into training/test portions
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0, stratify=y)
    
    # PCA part
    pca = PCA(n_components=2).fit(X)
    X_pca = pca.transform(X)
    
    pca = PCA(n_components=2).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    
    draw_crossval_graph(run_all_knn(X, y, X_train, y_train, X_test, y_test),
                        run_all_knn(X_pca, y, X_train_pca,y_train,X_test_pca,y_test),
                        list(range(1,k_neighbors,4)),
                        "balanced")
    
    
    
    
    
    
    
    
    
    
