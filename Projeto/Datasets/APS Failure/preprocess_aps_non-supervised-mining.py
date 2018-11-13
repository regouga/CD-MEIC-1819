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

#'class' is dropped because it is needed for classification, not unsupervised mining
#'ab_000', 'bm_000', 'bn_000','bo_000', 'bp_000', 'bq_000', 'br_000', 'cr_000' are dropped because they have >65% missing values
#
drop = ['class']


training_dataset = pd.read_csv("aps_failure_training_set.csv", sep=',', header=14, engine='python')
training_dataset = training_dataset.replace('na', -1)

for i in range(len(drop)):
    training_dataset = training_dataset.drop(drop[i], axis=1)
training_dataset.to_csv("base_aps_failure_training.csv", index=False)


test_dataset = pd.read_csv("aps_failure_test_set.csv", sep=',', header=14, engine='python')
for i in range(len(drop)):
    test_dataset = test_dataset.drop(drop[i], axis=1)
test_dataset.to_csv("base_aps_failure_test.csv", index=False)

training_dataset = training_dataset.apply(pd.to_numeric)




z = np.abs(stats.zscore(training_dataset))

training_dataset_o = training_dataset[(z < 3).all(axis=1)]



Q1 = training_dataset.quantile(0.25)
Q3 = training_dataset.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

training_dataset = training_dataset[((training_dataset < (Q1 - 1.5 * IQR)) |(training_dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
training_dataset.to_csv("treino_filtrado_outliers.csv", index=False)


training_dataset = training_dataset.replace(-1, np.nan)



columns = list(training_dataset.columns.values)

for e in columns:
    print(e, training_dataset[e].mean())
    training_dataset[e] = training_dataset[e].fillna(training_dataset[e].mean())

training_dataset.to_csv("treino_filtrado_outliers_media.csv", index=False)
    