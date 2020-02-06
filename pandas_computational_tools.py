#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:07:14 2020

@author: sadrachpierre
"""

import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


df_TSLA = pd.read_csv("TSLA.csv")

print(df_TSLA.head())

print(df_TSLA['Open'].pct_change().head())

print(df_TSLA[["Open", "High", "Low", "Close"]].pct_change().head())

print(df_TSLA[["Open", "High", "Low", "Close"]].pct_change(periods = 3).head(6))


df_AAPL = pd.read_csv("AAPL.csv")

Series_TSLA = df_TSLA['Open']
Series_AAPL = df_AAPL['Open']

print("Covariance Between TSLA and AAPL:", Series_TSLA.cov(Series_AAPL))


print(df_TSLA.cov().head())


print(df_AAPL.cov().head())



print("Correlation Between TSLA and AAPL:", Series_TSLA.corr(Series_AAPL))

print(df_TSLA.corr().head())

print(df_AAPL.corr().head())

print(df_AAPL['Open'].cumsum().head())

print(df_AAPL['Open'].rolling(window = 10).mean().head(20))

df_AAPL = df_AAPL[['Date', 'Open']]
df_AAPL['Date'] = pd.to_datetime(df_AAPL['Date'])
df_AAPL.set_index('Date', inplace = True)
df_AAPL = df_AAPL['Open']

print(df_AAPL.head())
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
plt.xlabel('Date')
plt.ylabel('Price')
df_AAPL.plot(style = 'k--')

df_AAPL.rolling(window = 10).mean().plot(style = 'k')