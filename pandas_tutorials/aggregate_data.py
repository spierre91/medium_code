#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:58:35 2020

@author: sadrachpierre
"""


import pandas as pd
df = pd.read_csv("winemag-data-130k-v2.csv")
print(len(df))
#df.fillna(0, inplace =True)
del df['Unnamed: 0']

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(df.head())

print(df['price'].describe())

print(df['points'].describe())


print(df['price'].groupby(df['variety']).mean().head())
print(df['price'].groupby(df['variety']).mean().sort_values(ascending = False).head())

print(df['price'].groupby(df['province']).mean().sort_values(ascending = False).head())

print(df[['price', 'points']].groupby(df.province).mean().head())#.head().sort_values(ascending = False))