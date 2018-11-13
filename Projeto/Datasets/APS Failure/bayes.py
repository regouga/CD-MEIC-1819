# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 19:23:03 2018

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
import seaborn as sns

data = pd.read_csv("base_aps_failure_trainingCla.csv")

GRAPHS_FOLDER = "bayes_graphs/"
X = np.array(data.drop("class",axis=1))
y = np.array(data["class"])
target_names = np.array(['neg','pos'])

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


oversampling = pd.DataFrame(np.column_stack((X_sm, y_sm)))
#oversampling.to_csv("samplingggg.csv", index=False)

pca = PCA(n_components=3).fit(X_sm)
X_sm_pca = pca.transform(X_sm)

pca = PCA(n_components=3).fit(X_train_sm)
X_train_sm_pca = pca.transform(X_train_sm)
X_test_sm_pca = pca.transform(X_test)


ada = ADASYN(random_state=2)
X_ada, y_ada = ada.fit_sample(X,y)
X_train_ada, y_train_ada = ada.fit_sample(X_train,y_train)

pca = PCA(n_components=3).fit(X_ada)
X_ada_pca = pca.transform(X_ada)

pca = PCA(n_components=3).fit(X_train_ada)
X_train_ada_pca = pca.transform(X_train_ada)
X_test_ada_pca = pca.transform(X_test)



def run_all_knn(X, y, X_train, y_train, X_test, y_test):
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("\nNaive Bayes classifier")
    print(classification_report(y_test,y_pred,target_names=target_names))
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred, labels=range(2)).ravel()
    print("TN: %d \tFP: %d \nFN: %d \tTP: %d" % (tn, fp, fn, tp))
    print("Accuracy score: %f" % (accuracy_score(y_test,y_pred)))
    print("ROC auc score: %f" % (roc_auc_score(y_test,y_pred)))
    clf = GaussianNB()
    clf.fit(X,y)
    print("Cross-Validation (10-fold) score: %f" % (cross_val_score(clf, X, y, cv=10).mean()))


def return_metric_vectors(metric, k,X_train, y_train, X_test, y_test, X_train_pca, X_test_pca):
    metrics_functions = {
        "roc_auc" : roc_auc_score,
        "accuracy" : accuracy_score,
        "precision" : precision_score,
        "recall" : recall_score 
    }
    metric_score = metrics_functions[metric]
    non_pca_metrics, with_pca_metrics = [], []
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    non_pca_metrics.append(metric_score(y_test,y_pred))

    clf = GaussianNB()
    clf.fit(X_train_pca,y_train)
    y_pred = clf.predict(X_test_pca)
    with_pca_metrics.append(metric_score(y_test,y_pred))

    return [non_pca_metrics,with_pca_metrics]


metrics_titles = {
    "roc_auc" : "AUC ROC score",
    "accuracy" : "Accuracy score",
    "precision" : "Precision score",
    "recall" : "Recall score"   
}


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
    plt.gca().set_ylim([0.46,1])
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

print("\n================= Basic Non-PCA =============================")
run_all_knn(X_pca, y, X_train_pca,y_train,X_test_pca,y_test)
print("===============================================================")

print("\n================= (OS) SMOTE Non-PCA ========================")
run_all_knn(X, y, X_train_sm, y_train_sm, X_test, y_test)
print("===============================================================")

print("\n================= (OS) ADASYN Non-PCA =======================")
run_all_knn(X, y, X_train_ada, y_train_ada, X_test, y_test)
print("===============================================================")

print("\n================ Basic PCA executions =======================")
run_all_knn(X_pca, y, X_train_pca,y_train,X_test_pca,y_test)
print("===============================================================")

print("\n================= (OS) SMOTE PCA ============================")
run_all_knn(X_sm_pca, y_sm, X_train_sm_pca,y_train_sm,X_test_sm_pca,y_test)
print("===============================================================")

print("\n================= (OS) ADASYN PCA ===========================")
run_all_knn(X_ada_pca, y_ada, X_train_ada_pca,y_train_ada,X_test_ada_pca,y_test)
print("===============================================================")
draw_learning_curve(X,y,X_pca,"default")
draw_learning_curve(X_sm,y_sm,X_sm_pca, "SMOTE")
draw_learning_curve(X_ada,y_ada, X_ada_pca, "ADASYN")

for n in range(1,11):
    sm = ADASYN(random_state=2)
    X_sm, y_sm = sm.fit_sample(X,y)
    X_train_sm, y_train_sm = sm.fit_sample(X_train,y_train)

    pca_test = PCA(n_components=n).fit(X_sm)
    X_sm_pca_test = pca_test.transform(X_sm)

    pca_test = PCA(n_components=n).fit(X_train_sm)
    X_train_sm_pca_test = pca_test.transform(X_train_sm)
    X_test_sm_pca_test = pca_test.transform(X_test)

    # pca_test = PCA(n_components=n).fit(X)
    # X_pca_test = pca_test.transform(X)

    # pca_test = PCA(n_components=n).fit(X_train)
    # X_train_pca_test = pca_test.transform(X_train)
    # X_test_pca_test = pca_test.transform(X_test)

    clf = GaussianNB()
    clf.fit(X_train_sm_pca_test,y_train_sm)
    y_pred = clf.predict(X_test_sm_pca_test)
    print("Accuracy score for %d components: %f" % (n , (accuracy_score(y_test,y_pred))))
