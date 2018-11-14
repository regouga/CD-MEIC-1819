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


for i in drops:
    green_dataset = green_dataset.drop(i, axis=1)
    hinselmann_dataset = hinselmann_dataset.drop(i, axis=1)
    schiller_dataset = schiller_dataset.drop(i, axis=1)

green_dataset.to_csv("base_QA-Green_unsupervised-mining.csv", index=False)
hinselmann_dataset.to_csv("base_QA-hinselmann_unsupervised-mining.csv", index=False)
schiller_dataset.to_csv("base_QA-schiller_unsupervised-mining.csv", index=False)