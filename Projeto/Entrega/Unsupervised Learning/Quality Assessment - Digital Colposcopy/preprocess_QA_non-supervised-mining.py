# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 22:19:10 2018

@author: João Pina
"""

import pandas as pd

drops = ['experts::0', 'experts::1', 'experts::2', 'experts::3', 'experts::4', 'experts::5', 'consensus']


green_dataset = pd.read_csv("green.csv", sep=',', engine='python')
hinselmann_dataset = pd.read_csv("hinselmann.csv", sep=',', engine='python')
schiller_dataset = pd.read_csv("schiller.csv", sep=',', engine='python')


dataset = green_dataset.append(hinselmann_dataset)
dataset = dataset.append(schiller_dataset)


for i in drops:
    dataset = dataset.drop(i, axis=1)


dataset.to_csv("base_QA_unsupervised-mining.csv", index=False)
