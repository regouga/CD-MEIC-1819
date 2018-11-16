# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 22:01:01 2018

@author: João Pina
"""


import pandas as pd, numpy as np
from IPython.display import display, HTML
from sklearn.preprocessing import LabelBinarizer
from mlxtend.frequent_patterns import apriori, association_rules



iondata = pd.read_csv("ianosphere.csv", sep=',', engine='python')
attrs = []
for attr in list(iondata): attrs.append(attr)
print(attrs)
#df = pd.DataFrame(data=iondata['data'], columns=attrs)
#with pd.option_context('display.max_rows', 10, 'display.max_columns', 8): print(df)


for col in list(df) :
    if col not in ['class','a01','a02'] :
        df[col] = pd.cut(df[col],3,labels=['0','1','2'])
    attrs = []
    values = df[col].unique().tolist()
    values.sort()
    for val in values : attrs.append("%s:%s"%(col,val))
    lb = LabelBinarizer().fit_transform(df[col])
    if(len(attrs)==2) :
        v = list(map(lambda x: 1 - x, lb))
        lb = np.concatenate((lb,v),1)
    df2 = pd.DataFrame(data=lb, columns=attrs)
    df = df.drop(columns=[col])
    df = pd.concat([df,df2], axis=1, join='inner')
with pd.option_context('display.max_rows', 10, 'display.max_columns', 8): print(df)
frequent_itemsets = apriori(df, min_support=0.7, use_colnames=True)
#print(frequent_itemsets)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
display(HTML(rules.to_html()))