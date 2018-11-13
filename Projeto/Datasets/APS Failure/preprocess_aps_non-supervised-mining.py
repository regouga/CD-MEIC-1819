# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 16:47:59 2018

@author: João Pina
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy import stats
import numpy as np




training_dataset = pd.read_csv("aps_failure_training_set.csv", sep=',', header=14, engine='python')
test_dataset = pd.read_csv("aps_failure_test_set.csv", sep=',', header=14, engine='python')

dataset = training_dataset.append(test_dataset)

dataset = dataset.drop('class', axis=1)

dataset = dataset.replace('na', -1)

dataset = dataset.apply(pd.to_numeric)

z = np.abs(stats.zscore(dataset))

print(z)

dataset_o = dataset[(z < 3).all(axis=1)]



Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

dataset = dataset[((dataset < (Q1 - 1.5 * IQR)) |(dataset > (Q3 + 1.5 * IQR))).any(axis=1)]


dataset = dataset.replace(-1, np.nan)



columns = list(dataset.columns.values)

for e in columns:
    print(e, dataset[e].mean())
    dataset[e] = dataset[e].fillna(dataset[e].mean())

dataset.to_csv("base_aps-failure_unsupervised-mining.csv", index=False)