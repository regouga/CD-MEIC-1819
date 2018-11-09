# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 22:19:10 2018

@author: João Pina
"""

import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("green.csv", sep=',', engine='python')

columns = list(dataset.columns.values)

for i in range(len(columns)):
    print("COLUMN: ", columns[i])
    dataset.hist(column = columns[i])