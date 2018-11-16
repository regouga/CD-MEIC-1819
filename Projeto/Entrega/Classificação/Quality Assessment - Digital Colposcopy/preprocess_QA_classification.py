# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 22:19:10 2018

@author: Miguel Regouga
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy import stats
import numpy as np
import seaborn as sns

green_dataset = pd.read_csv("green.csv", sep=',', engine='python')
hinselmann_dataset = pd.read_csv("hinselmann.csv", sep=',', engine='python')
schiller_dataset = pd.read_csv("schiller.csv", sep=',', engine='python')

'''
zA = np.abs(stats.zscore(green_dataset))
green_dataset = green_dataset[(zA < 3).all(axis=1)]
Q1A = green_dataset.quantile(0.25)
Q3A = green_dataset.quantile(0.75)
IQRA = Q3A - Q1A
green_dataset = green_dataset[((green_dataset < (Q1A - 1.5 * IQRA)) |(green_dataset > (Q3A + 1.5 * IQRA))).any(axis=1)]


green_col = list(green_dataset)
green_col.remove('consensus')
green_col.remove('experts::5')
green_col.remove('experts::4')
green_col.remove('experts::3')
green_col.remove('experts::2')
green_col.remove('experts::1')
green_col.remove('experts::0')
print(green_col)
#for col in green_col:
#    a = sns.boxplot(x=green_dataset[col])






zB = np.abs(stats.zscore(hinselmann_dataset))
hinselmann_dataset = hinselmann_dataset[(zB < 3).all(axis=1)]
Q1B = hinselmann_dataset.quantile(0.25)
Q3B = hinselmann_dataset.quantile(0.75)
IQRB = Q3B - Q1B
hinselmann_dataset = hinselmann_dataset[((hinselmann_dataset < (Q1B - 1.5 * IQRB)) |(hinselmann_dataset > (Q3B + 1.5 * IQRB))).any(axis=1)]

zC = np.abs(stats.zscore(schiller_dataset))
schiller_dataset = schiller_dataset[(zC < 3).all(axis=1)]
Q1C = schiller_dataset.quantile(0.25)
Q3C = schiller_dataset.quantile(0.75)
IQRC = Q3C - Q1C
schiller_dataset = schiller_dataset[((schiller_dataset < (Q1C - 1.5 * IQRC)) |(schiller_dataset > (Q3C + 1.5 * IQRC))).any(axis=1)]
'''

frames = [green_dataset, hinselmann_dataset, schiller_dataset]

result = pd.concat(frames)
result.to_csv("col_classification.csv", index=False)
