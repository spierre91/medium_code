#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 18:45:46 2019

@author: sadrachpierre
"""

import pandas as pd 

df = pd.read_csv("top50.csv", encoding="ISO-8859-1")

print(df.head())


df.sort_values('Popularity', ascending = False, inplace = True)

print(df.head())

import matplotlib.pyplot as plt

from collections import Counter

print(Counter(df['Artist.Name'].values))
print(dict(Counter(df['Artist.Name'].values).most_common(5)))
bar_plot = dict(Counter(df['Artist.Name'].values).most_common(5))
plt.bar(*zip(*bar_plot.items()))
plt.show()



def get_frequencies(column_name):
    print(Counter(df[column_name].values))
    print(dict(Counter(df[column_name].values).most_common(5)))
    bar_plot = dict(Counter(df[column_name].values).most_common(5))
    plt.bar(*zip(*bar_plot.items()))
    plt.show()


get_frequencies('Genre')

get_frequencies('Artist.Name')