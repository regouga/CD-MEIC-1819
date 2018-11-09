# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 16:47:59 2018

@author: João Pina
"""

import pandas as pd
import missingno as msno

training_dataset = pd.read_csv("aps_failure_training_set.csv", sep=',', header=14, engine='python')

training_dataset['class'] = training_dataset['class'].replace('neg', 0)
training_dataset['class'] = training_dataset['class'].replace('pos', 1)

training_dataset.drop('ab_000', axis=1)
training_dataset.drop('bm_000', axis=1)
training_dataset.drop('bn_000', axis=1)
training_dataset.drop('bo_000', axis=1)
training_dataset.drop('bp_000', axis=1)
training_dataset.drop('bq_000', axis=1)
training_dataset.drop('br_000', axis=1)
training_dataset.drop('cr_000', axis=1)

training_dataset.to_csv("base_aps_failure_training.csv", index=False)


test_dataset = pd.read_csv("aps_failure_test_set.csv", sep=',', header=14, engine='python')

test_dataset['class'] = test_dataset['class'].replace('neg', 0)
test_dataset['class'] = test_dataset['class'].replace('pos', 1)


test_dataset.drop('ab_000', axis=1)
test_dataset.drop('bm_000', axis=1)
test_dataset.drop('bn_000', axis=1)
test_dataset.drop('bo_000', axis=1)
test_dataset.drop('bp_000', axis=1)
test_dataset.drop('bq_000', axis=1)
test_dataset.drop('br_000', axis=1)
test_dataset.drop('cr_000', axis=1)

test_dataset.to_csv("base_aps_failure_test.csv", index=False)



