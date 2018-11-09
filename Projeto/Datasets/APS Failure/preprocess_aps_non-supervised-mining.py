# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 16:47:59 2018

@author: João Pina
"""

import pandas as pd
#'class' is dropped because it is needed for classification, not unsupervised mining
#'ab_000', 'bm_000', 'bn_000','bo_000', 'bp_000', 'bq_000', 'br_000', 'cr_000' are dropped because they have >65% missing values
#
drop = ['class']




training_dataset = pd.read_csv("aps_failure_training_set.csv", sep=',', header=14, engine='python')

#import seaborn as sns
#sns.boxplot(x=training_dataset['ad_000'])


for i in range(len(drop)):
    training_dataset = training_dataset.drop(drop[i], axis=1)
training_dataset.to_csv("base_aps_failure_training.csv", index=False)


test_dataset = pd.read_csv("aps_failure_test_set.csv", sep=',', header=14, engine='python')
for i in range(len(drop)):
    test_dataset = test_dataset.drop(drop[i], axis=1)
test_dataset.to_csv("base_aps_failure_test.csv", index=False)