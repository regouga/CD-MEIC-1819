# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 22:36:51 2018

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

data = pd.read_csv("base_aps_failure_trainingCla.csv")

GRAPHS_FOLDER = "bayes_graphs/"
X = np.array(data.drop("class",axis=1))
y = np.array(data["class"])
labels = pd.unique(y)

# split dataset into training/test portions
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0, stratify=y)

# PCA part
pca = PCA(n_components=3).fit(X)
X_pca = pca.transform(X)

pca = PCA(n_components=3).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Over-sampling techniques
sm = SMOTE(random_state=1)
X_sm, y_sm = sm.fit_sample(X,y)
X_train_sm, y_train_sm = sm.fit_sample(X_train,y_train)

pca = PCA(n_components=3).fit(X_sm)
X_sm_pca = pca.transform(X_sm)

pca = PCA(n_components=3).fit(X_train_sm)
X_train_sm_pca = pca.transform(X_train_sm)
X_test_sm_pca = pca.transform(X_test)

def draw_learning_curve(X, y, X_pca, filename):
    clf = GaussianNB()
    train_sizes,train_scores, test_scores = learning_curve(
        clf, X, y, cv=10, n_jobs=8)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    clf_pca = GaussianNB()
    train_sizes_pca,train_scores_pca, test_scores_pca = learning_curve(
        clf, X_pca, y, cv=10, n_jobs=8)
    train_scores_mean_pca = np.mean(train_scores_pca, axis=1)
    train_scores_std_pca = np.std(train_scores_pca, axis=1)
    test_scores_mean_pca = np.mean(test_scores_pca, axis=1)
    test_scores_std_pca = np.std(test_scores_pca, axis=1)

    f = plt.figure()
    plt.title("Learning Curve Naive Bayes - " + filename)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().set_ylim([0.46,0.86])
    plt.grid()

    plt.plot(train_sizes, train_scores_mean, '.-', color="r",
             label="Training score Non-PCA")
    plt.plot(train_sizes_pca, train_scores_mean_pca, '.-', color="b",
             label="Training score with PCA")
    plt.plot(train_sizes, test_scores_mean, '.-', color="g",
             label="Cross-validation score Non-PCA")
    plt.plot(train_sizes_pca, test_scores_mean_pca, '.-', color="y",
             label="Cross-validation score with PCA")

    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))   
    #plt.show()
    f.savefig(GRAPHS_FOLDER+"lc_bayes_"+filename+".png",bbox_inches="tight")
    f.savefig(GRAPHS_FOLDER+"lc_bayes_"+filename+".pdf",bbox_inches="tight")


def run_all_knn(X, y, X_train, y_train, X_test, y_test):
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("\nNaive Bayes classifier")
    print(classification_report(y_test,y_pred,target_names=labels))
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred, labels=range(2)).ravel()
    print("TN: %d \tFP: %d \nFN: %d \tTP: %d" % (tn, fp, fn, tp))
    print("Accuracy score: %f" % (accuracy_score(y_test,y_pred)))
    print("ROC auc score: %f" % (roc_auc_score(y_test,y_pred)))
    clf = GaussianNB()
    clf.fit(X,y)
    print("Cross-Validation (10-fold) score: %f" % (cross_val_score(clf, X, y, cv=10).mean()))
    
def run_all_knn(X, y, X_train, y_train, X_test, y_test):
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("\nKNN classifier with %d neighbors" % (n))
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred, labels=range(2)).ravel()
    print("TN: %d \tFP: %d \nFN: %d \tTP: %d" % (tn, fp, fn, tp))
    print("Accuracy score: %f" % (accuracy_score(y_test,y_pred)))
    print("ROC auc score: %f" % (roc_auc_score(y_test,y_pred)))
    #clf = GaussianNB()
    #clf.fit(X,y)
    #print("Cross-Validation (10-fold) score: %f" % (cross_val_score(clf, X, y, cv=10).mean()))
    #cross_val_vector.append(cross_val_score(clf, X, y, cv=10).mean())
    
    return (cross_val_vector, 
            precision_score(y_test, y_pred, labels=labels, pos_label=1, average='binary', sample_weight=None)[source],
            recall_score(y_test, y_pred, labels=labels, pos_label=1, average='binary', sample_weight=None))

def draw_graph(res1, res2, k_values, balance):
    #cross_val_vector = res1[0]
    #cross_val_vector_pca = res2[0]
    precision = res1[1]
    precision_pca = res2[1]
    recall = res1[2]
    recall_pca = res2[2]
    f = plt.figure()
    	
    plt.title("precision/recall score vs k-neighbors with %s data" %(balance))
    plt.xlabel("k-neighbors")
    plt.ylabel("precision/recall score")
    plt.gca().set_ylim([0,1])
    plt.grid()
    
    plt.plot(k_values, precision, '.-', color="r", label="precision Non-PCA")
    plt.plot(k_values, precision_pca, '.-', color="b", label="precision with PCA")
    
    plt.plot(k_values, recall, '.-', color="g", label="recall Non-PCA")
    plt.plot(k_values, recall_pca, '.-', color="y", label="recall with PCA")
    	
    plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))	
    
    f.savefig("%s%s_knn_%s.png" % (GRAPHS_FOLDER,"crossval",balance),bbox_inches="tight")
    f.savefig("%s%s_knn_%s.pdf" % (GRAPHS_FOLDER,"crossval",balance),bbox_inches="tight")

    
print("\n================= Basic Non-PCA =============================")
run_all_knn(X, y, X_train, y_train, X_test, y_test)
print("===============================================================")

print("\n================= (OS) SMOTE Non-PCA ========================")
run_all_knn(X, y, X_train_sm, y_train_sm, X_test, y_test)
print("===============================================================")

print("\n================ Basic PCA executions =======================")
run_all_knn(X_pca, y, X_train_pca,y_train,X_test_pca,y_test)
print("===============================================================")

print("\n================= (OS) SMOTE PCA ============================")
run_all_knn(X_sm_pca, y_sm, X_train_sm_pca,y_train_sm,X_test_sm_pca,y_test)
print("===============================================================")

draw_learning_curve(X,y,X_pca,"default")
draw_learning_curve(X_sm,y_sm,X_sm_pca, "SMOTE")