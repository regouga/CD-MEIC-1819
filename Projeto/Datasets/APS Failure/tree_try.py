# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 22:11:32 2018

@author: Hélio
"""


import pandas as pd
import numpy as np
import graphviz 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
data = pd.read_csv("base_aps_failure_trainingCla.csv")
X = np.array(data.drop("class",axis=1))
y = np.array(data["class"])
target_names = np.array(['0','1'])
#feature_names = data.axes[1][1:]
feature_names = list(data)[1:]
print(feature_names)
teste = pd.read_csv("base_aps_failure_testCla.csv")
X_test = np.array(data.drop("class",axis=1))
y_test = np.array(data["class"])

# split dataset into training/test portions



def decisionTree(X, X_train, y_train, X_test, y_test, min_sample_leaf, min_sample_node):
	
    clf = DecisionTreeClassifier()
    
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    dot_data = tree.export_graphviz(clf, out_file=None,feature_names=feature_names,class_names=target_names,  filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("dnmkc", view=True)
    treeObj = clf.tree_
    print(treeObj.node_count, accuracy_score(y_test,y_pred))
    #return str(treeObj.node_count) +","+ str(accuracy_score(y_test,y_pred))


decisionTree(X, X, y, X_test, y_test, 1,2)
# for i in range(1,51,12):
# 	some =""
# 	for j in range(2, 51, 12):
# 		some+=str(j)+","+str(i)+","+decisionTree(X, X_train, y_train, X_test, y_test, i, j)+","
# 	print(some[:-1])


# for i in range(101,2,-1):
# 	print(decisionTree(X, X_train, y_train, X_test, y_test, 1, i))	

# print(decisionTree(X, X_train, y_train, X_test, y_test, 1, 90))
# print(decisionTree(X, X_train, y_train, X_test, y_test, 1, 50))

# clf = DecisionTreeClassifier()
# train_sizes,train_scores, test_scores = learning_curve(
#     clf, X, y, cv=10, n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)


# plt.figure()
# plt.title("Learning Curve DT's")
# plt.xlabel("Training examples")
# plt.ylabel("Score")
# plt.grid()

# plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.1,
#                  color="r")
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#          label="Training score")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#          label="Cross-validation score")

# plt.legend(loc="best")
# plt.show()