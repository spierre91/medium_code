#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:16:23 2019

@author: sadrachpierre
"""

import pandas as pd

df = pd.read_csv("debris_flow_data.csv")
df = df[['StormDur_H', 'StormAccum_mm', 'StormAvgI_mm/h',
         'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h',
         'PropHM23', 'dNBR/1000', 'KF', 'Acc015_mm', 'Acc030_mm',
         'Acc060_mm', 'Response']]
from collections import Counter
print(Counter(df['Response'].values))

df_DF = df[df['Response'] == 1]
df_nDF = df[df['Response'] == 0]
df_nDF = df_nDF.sample(n=len(df_DF))
df = df_DF.append(df_nDF)

df.dropna(inplace = True)

import numpy as np 
X = np.array(df[['StormDur_H', 'StormAccum_mm', 'StormAvgI_mm/h',
         'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h',
         'PropHM23', 'dNBR/1000', 'KF', 'Acc015_mm', 'Acc030_mm',
         'Acc060_mm']])
y = np.array(df['Response'])

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

kf = KFold(n_splits=10, random_state=42, shuffle=True)
mean_error = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = RandomForestClassifier(n_estimators = 100, max_depth = 100, random_state = 42)
    model.fit(X_train, y_train)
    predictions = model.predict(np.array(X_test))
    mean_error.append(roc_auc_score(y_test, predictions))
    print("Accuracy:", roc_auc_score(y_test, predictions))
    
print("Mean Error: ", sum(mean_error)/len(mean_error))
