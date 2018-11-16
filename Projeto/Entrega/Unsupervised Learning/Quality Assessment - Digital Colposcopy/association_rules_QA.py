# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 19:13:10 2018

@author: João Pina
"""

import pandas as pd, numpy as np
from sklearn.preprocessing import LabelBinarizer
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

dataset = pd.read_csv("base_QA_unsupervised-mining.csv", sep=',', engine='python')
dataset.head()

for col in list(dataset):
    dataset[col] = pd.cut(dataset[col],9,labels=['0','1','2','3','4','5','6','7','8'])
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

support = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

nr_regras = []
lifts=[]


for i in support:
    frequent_itemsets = apriori(dataset, min_support=i, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    rules = rules[rules.lift > 1.5]
    fileName = "9_bins/QA_association_rules_bins=9_sup=" + str(i) + "_conf=0.5.csv"
    rules.to_csv(fileName, index=False)
    nr_regras = nr_regras + [len(rules)]
    lifts += [rules.lift.mean()]
    

nr_regras_9 = [108922, 8280, 162, 0, 0, 0, 0]
nr_regras_10 = [63689, 6524, 18, 0, 0, 0, 0]
nr_regras_11 = [25149, 2206, 0, 0, 0, 0, 0]
        
plt.plot(support, nr_regras)
plt.xlabel('Support')
plt.ylabel('Nº de Regras')
plt.title('Variação do nº de regras com o support - Quality Assurance Dataset (9 bins)')
plt.show()

lifts_9 = [1.722469299454125, 1.6499519268970273, 1.5790493347758952, 0, 0, 0, 0]
lifts_10 = [1.812976664084475, 1.7022643152957304, 1.6654258039919125, 0, 0, 0, 0]
lifts_11 = [1.7321760474569865, 1.7688167366120544, 0, 0, 0, 0, 0]


plt.plot(support, nr_regras_9, '.-', color="r", label="9 bins")
plt.plot(support, nr_regras_10, '.-', color="b", label="10 bins")
plt.plot(support, nr_regras_11, '.-', color="c", label="11 bins")
plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))	
plt.xlabel('Support')
plt.ylabel('Nº de Regras')
plt.title('Variação do nº de regras com o support - Quality Assurance Dataset')
plt.show()


plt.plot(support, lifts_9, '.-', color="r", label="9 bins")
plt.plot(support, lifts_10, '.-', color="b", label="10 bins")
plt.plot(support, lifts_11, '.-', color="c", label="11 bins")
plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))	
plt.xlabel('Support')
plt.ylabel('Lift')
plt.title('Variação do Lift com o support - Quality Assurance Dataset')
plt.show()        
    