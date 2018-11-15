# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 23:22:11 2018

@author: Hélio
"""

from sklearn.decomposition import PCA
from sklearn.metrics import *
from sklearn.tree import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
#import graphviz 
from math import *
import os
import matplotlib.pyplot as plt

GRAPHS_FOLDER = "tree/"
data = pd.read_csv("base_aps_failure_trainingCla.csv")
X = np.array(data.drop("class",axis=1))
y = np.array(data["class"])
target_names = np.array([0,1])
feature_names = np.array(["aa_000","ab_000","ax_000"])

teste = pd.read_csv("base_aps_failure_testCla.csv")
X_test = np.array(data)

import pydotplus
import collections
clf = DecisionTreeClassifier(min_samples_leaf=1, min_samples_split=2)
clf = clf.fit(X,y)
y_pred = clf.predict(X_test)

dot_data = tree.export_graphviz(clf,feature_names=features_list, out_file=None,filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')