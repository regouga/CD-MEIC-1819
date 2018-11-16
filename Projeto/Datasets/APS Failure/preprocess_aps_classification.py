# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 16:47:59 2018

@author: João Pina
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy import stats
import numpy as np
from imblearn.over_sampling import SMOTE

training_dataset = pd.read_csv("aps_failure_training_set.csv", sep=',', header=14, engine='python')

training_dataset.to_csv("base_aps_failure_trainingCla.csv", index=False)


test_dataset = pd.read_csv("aps_failure_test_set.csv", sep=',', header=14, engine='python')

training_dataset = training_dataset.replace('neg', 0)
training_dataset = training_dataset.replace('pos', 1)
test_dataset = test_dataset.replace('neg', 0)
test_dataset = test_dataset.replace('pos', 1)

training_dataset = training_dataset.replace('na', -1)
test_dataset = test_dataset.replace('na', -1)
training_dataset = training_dataset.apply(pd.to_numeric)
test_dataset = test_dataset.apply(pd.to_numeric)
'''
z1 = np.abs(stats.zscore(training_dataset))
z2 = np.abs(stats.zscore(test_dataset))

training_dataset_o = training_dataset[(z1 < 3).all(axis=1)]
test_dataset_o = test_dataset[(z2 < 3).all(axis=1)]

Q1 = training_dataset.quantile(0.25)
Q3 = training_dataset.quantile(0.75)
IQR = Q3 - Q1

Q11 = test_dataset.quantile(0.25)
Q33 = test_dataset.quantile(0.75)
IQR1 = Q33 - Q11

training_dataset = training_dataset[((training_dataset < (Q1 - 1.5 * IQR)) |(training_dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
test_dataset = test_dataset[((test_dataset < (Q11 - 1.5 * IQR1)) |(test_dataset > (Q33 + 1.5 * IQR1))).any(axis=1)]
'''
training_dataset = training_dataset.replace(-1, np.nan)
test_dataset = test_dataset.replace(-1, np.nan)


columns = list(training_dataset.columns.values)

for e in columns:
    training_dataset[e] = training_dataset[e].fillna(training_dataset[e].mean())
    test_dataset[e] = test_dataset[e].fillna(test_dataset[e].mean())

###BALANCEAR COM SMOTE
#sm = SMOTE(random_state=1)
#X = np.array(training_dataset.drop("class",axis=1))
#y = np.array(training_dataset["class"])
#X, y = sm.fit_sample(X, y)

training_dataset.to_csv("base_aps_failure_trainingCla.csv", index=False)
test_dataset.to_csv("base_aps_failure_testCla.csv", index=False)


