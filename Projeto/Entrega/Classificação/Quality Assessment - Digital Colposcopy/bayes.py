# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 20:42:11 2018

@author: Hélio
"""

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
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

GRAPHS_FOLDER = "bayes/"
files = ["green.csv", "schiller.csv", "hinselmann.csv", "col_classification.csv"]
for file in files:
    print("=================================", file)
    data = pd.read_csv(file)
    exp = ["experts::0","experts::1","experts::2","experts::3","experts::4","experts::5","consensus"]
    # using n_jobs to try and paralelize computation when possible!
    def run_all_knn(X, y, X_train, y_train, X_test, y_test):
        clf = GaussianNB()
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        #print("\nNaive Bayes classifier")
        #print(classification_report(y_test,y_pred,target_names=target_names))
        tn, fp, fn, tp = confusion_matrix(y_test,y_pred, labels=range(2)).ravel()
        #print("TN: %d \tFP: %d \nFN: %d \tTP: %d" % (tn, fp, fn, tp))
        #print("Accuracy score: %f" % (accuracy_score(y_test,y_pred)))
        #print("ROC auc score: %f" % (roc_auc_score(y_test,y_pred)))
        clf = GaussianNB()
        clf.fit(X,y)
        #print("Cross-Validation (10-fold) score: %f" % (cross_val_score(clf, X, y, cv=10).mean()))
        return (cross_val_score(clf, X, y, cv=10).mean(),accuracy_score(y_test,y_pred) )
    
    acc_non = []
    cross_non = []
    acc = []
    cross = []
    for i in exp:
        
        X = np.array(data.
                     drop("experts::0",axis=1).
                     drop("experts::1",axis=1).
                     drop("experts::2",axis=1).
                     drop("experts::3",axis=1).
                     drop("experts::4",axis=1).
                     drop("experts::5",axis=1).
                     drop("consensus",axis=1))
        y = np.array(data[i])
        target_names = np.array(["0","1"])

        # split dataset into training/test portions
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
        
        # PCA part
        pca = PCA(n_components=3).fit(X)
        X_pca = pca.transform(X)
        
        pca = PCA(n_components=3).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        res1 = run_all_knn(X, y, X_train, y_train, X_test, y_test)
        res2 = run_all_knn(X_pca, y, X_train_pca,y_train,X_test_pca,y_test)
        acc_non.append(res1[1])
        cross_non.append(res1[0])
        acc.append(res2[1])
        cross.append(res2[0])
    
    
    fig, ax = plt.subplots()
    
    ind = np.arange(len(exp))    # the x locations for the groups
    width = 0.2        # the width of the bars
    p1 = ax.bar(ind, acc, width, color='r')
    p2 = ax.bar(ind + width, acc_non, width,color='y')
    p3 = ax.bar(ind+ width+ width, cross, width, color='b')
    p4 = ax.bar(ind + width+ width+ width, cross_non, width,color='g')
    
    ax.set_title('accuracy & cross-validation')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(exp)
    ax.set_ylim([0.4,1])
    plt.show()
    fig.savefig("%sbayes_%s.png" % (GRAPHS_FOLDER,file),bbox_inches="tight")

    '''acc = []
    for n in range(2,11):
    
    
        pca_test = PCA(n_components=n).fit(X)
        X_pca_test = pca_test.transform(X)
    
        pca_test = PCA(n_components=n).fit(X_train)
        X_train_pca_test = pca_test.transform(X_train)
        X_test_pca_test = pca_test.transform(X_test)
    
        # pca_test = PCA(n_components=n).fit(X)
        # X_pca_test = pca_test.transform(X)
    
        # pca_test = PCA(n_components=n).fit(X_train)
        # X_train_pca_test = pca_test.transform(X_train)
        # X_test_pca_test = pca_test.transform(X_test)
    
        clf = GaussianNB()
        clf.fit(X_train_pca_test,y_train)
        y_pred = clf.predict(X_test_pca_test)
        accu = accuracy_score(y_test,y_pred)
        acc.append(accu)
        print("Accuracy score for %d components: %f" % (n , (accu)))
    print(acc)
    x = [2,3,4,5,6,7,8,9,10]
    x = pd.Series.from_array(x)
    width = 1/1.5
    plt.bar(x, acc, width, color="blue")
    plt.title("accuracy")
    plt.gca().set_ylim([0,1])
    plt.xlabel('componentes')
    plt.ylabel('accuracy score')
    fig = plt.gcf()
    plotly_fig = tls.mpl_to_plotly(fig)
    py.iplot(plotly_fig, filename='mpl-basic-bar')
    '''