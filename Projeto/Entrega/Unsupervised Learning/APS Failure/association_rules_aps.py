# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 19:13:10 2018

@author: João Pina
"""

import pandas as pd, numpy as np
from sklearn.preprocessing import LabelBinarizer
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

dataset = pd.read_csv("base_aps-failure_unsupervised-mining_with_class.csv", sep=',', engine='python')
dataset.head()

class_column = dataset['class']

dataset = dataset.drop('class', axis=1)

dataset.shape
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(dataset, class_column)
clf.feature_importances_  
model = SelectFromModel(clf, prefit=True)
feature_idx = model.get_support()
feature_name = dataset.columns[feature_idx]
dataset = model.transform(dataset)
dataset.shape 

dataset = pd.DataFrame(dataset, columns = feature_name)

for col in list(dataset) :
    dataset[col] = pd.cut(dataset[col],11,labels=['0','1','2','3','4','5','6','7','8','9','10'])
    attrs = []
    values = dataset[col].unique()
    values.sort_values()
    for val in values : attrs.append("%s:%s"%(col,val))
    lb = LabelBinarizer().fit_transform(dataset[col])
    if(len(attrs)==2) :
        v = list(map(lambda x: 1 - x, lb))
        lb = np.concatenate((lb,v),1)
    dataset2 = pd.DataFrame(data=lb, columns=attrs)
    dataset = dataset.drop(col, axis=1)
    dataset = pd.concat([dataset,dataset2], axis=1, join='inner')
with pd.option_context('display.max_rows', 10, 'display.max_columns', 8): print(dataset)

support = [0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9]

nr_regras = []
lifts = []

for i in support:
    frequent_itemsets = apriori(dataset, min_support=i, use_colnames=True, max_len=2)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    rules = rules[rules.lift > 1]
    fileName = "11_bins/APS_association_rules_bins=11_sup=" + str(i) + "_conf=0.5.csv"
    rules.to_csv(fileName, index=False)
    nr_regras = nr_regras + [len(rules)]
    lifts += [rules.lift.mean()]
        
plt.plot(support, nr_regras)
plt.xlabel('Support')
plt.ylabel('Nº Regras')
plt.title('Variação do nº de regras com o support - APS Failure Dataset (11 bins)')
plt.show()

nr_regras_9 = [2816, 2816, 2816, 2816, 2816, 2712, 2712]
nr_regras_10 = [3955, 3893, 3834, 3834, 3712, 3712, 3712]
nr_regras_11 = [3654, 3594, 3594, 3478, 3362, 3362, 3362]


lifts_11 = [1.0065569189350718, 1.006570595008961, 1.006570595008961, 1.0066000299301063, 1.0066000299301063, 1.0066000299301063, 1.0066000299301063]
lifts_10 = [1.0057665663182207, 1.0058030306155235, 1.0058406017915158, 1.0058406017915158, 1.0058406017915158, 1.0058406017915158, 1.0058406017915158]
lifts_9  = [1.0051438935936552, 1.0051438935936552, 1.0051438935936552, 1.005125234233428, 1.005125234233428, 1.0049923950012565, 1.0049923950012565]


plt.plot(support, nr_regras_9, '.-', color="r", label="9 bins")
plt.plot(support, nr_regras_10, '.-', color="b", label="10 bins")
plt.plot(support, nr_regras_11, '.-', color="c", label="11 bins")
plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))	
plt.xlabel('Support')
plt.ylabel('Nº de Regras')
plt.title('Variação do nº de regras com o support - APS Failure Dataset')
plt.show()


plt.plot(support, lifts_9, '.-', color="r", label="9 bins")
plt.plot(support, lifts_10, '.-', color="b", label="10 bins")
plt.plot(support, lifts_11, '.-', color="c", label="11 bins")
plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))	
plt.xlabel('Support')
plt.ylabel('Lift')
plt.title('Variação do Lift com o support - APS Failure Dataset')
plt.show()    
    