#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:51:27 2019

@author: sadrachpierre
"""
import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv("BrentOilPRices.csv")
df['Date'] = pd.to_datetime(df['Date'])
print(df.tail())

import seaborn as sns
sns.set()
plt.title('Brent Oil Prices')
sns.lineplot(df['Date'], df['Price'])

df['Price'].hist(bins = 100)