#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:27:22 2020

@author: sadrachpierre
"""

import pandas as pd 

df = pd.read_csv("FertilizersProduct.csv",encoding = "ISO-8859-1")

print(df.info())

print(df.head())

df_afghanistan = df[df['Area'] == 'Afghanistan'].copy()

print(df_afghanistan.head())

df_afghanistan = df_afghanistan[df_afghanistan['Unit'] == 'tonnes']

import matplotlib.pyplot as plt
import seaborn as sns 

sns.set()

plt.title("Distribution in Imported Fertilizer Quantities")
df_afghanistan['Value'].hist(bins=30)
print("Average Quantity: ", df_afghanistan['Value'].mean())
print("Standard Deviation in Quantity: ", df_afghanistan['Value'].std())
print("Correlation between Year and Quantity: ", df_afghanistan['Value'].corr(df_afghanistan['Year']))
print("Minimum Quantity: ", df_afghanistan['Value'].min())
print("Maximum Quantity: ", df_afghanistan['Value'].max())

print("Mode in Area: ", df_afghanistan['Area'].mode()[0])


from collections import Counter

print("Mode in Area: ", df['Area'].mode()[0])
from collections import Counter
print(Counter(df['Item']))

plt.title("Frequency in Fertilizer Items")
plt.xticks(rotation = 90)
plt.bar(dict(Counter(df['Item'])).keys(), dict(Counter(df['Item'])).values())