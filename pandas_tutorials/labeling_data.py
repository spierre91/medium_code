#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 16:12:53 2020

@author: sadrachpierre
"""

import pandas as pd 
import matplotlib.pyplot as plt
from collections import Counter

df_wine = pd.read_csv("winequality-red.csv")

print(df_wine.head())

print("Quality values: ", set(df_wine['quality']))

print("Max Alcohol %: ", df_wine['alcohol'].max())
print("Min Alcohol %: ", df_wine['alcohol'].min())

plt.title("Distribution in Alcohol %")
df_wine['alcohol'].hist()

import numpy as np
df_wine['alcohol_class'] = np.where(df_wine['alcohol']>=10.0, '1', '0')



plt.title("Distribution in Alcohol Class Labels")
plt.bar(dict(Counter(df_wine['alcohol_class'])).keys(), dict(Counter(df_wine['alcohol_class'])).values())

print("Max fixed acidity %: ", df_wine['fixed acidity'].max())
print("Min fixed acidity %: ", df_wine['fixed acidity'].min())

df_wine.loc[(df_wine['fixed acidity']>4.0) & (df_wine['fixed acidity']<=7.0), 'acidity_class'] = '0'
df_wine.loc[(df_wine['fixed acidity']>7.0) & (df_wine['fixed acidity']<=9.0), 'acidity_class'] = '1'
df_wine.loc[(df_wine['fixed acidity']>9.0) & (df_wine['fixed acidity']<=16.0), 'acidity_class'] = '2'



print(df_wine.head(100))
plt.title("Distribution in Acidity Class Labels")
plt.bar(dict(Counter(df_wine['acidity_class'])).keys(), dict(Counter(df_wine['acidity_class'])).values())