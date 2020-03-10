#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:05:44 2020

@author: sadrachpierre
"""

import pandas as pd 

df = pd.read_clipboard()
print(df.head())
df.to_csv("dow_jones.csv")
df = pd.read_csv("dow_jones.csv")
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
del df['Unnamed: 0']
df['Date'] = pd.to_datetime(df['Date'])

df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.week

df = df.query('Month == 3')

df['Open'] = df['Open'].str.replace(',', '')
df['Open'] = df['Open'].astype(float)
df['Open^2'] = df['Open'].apply(lambda x:x**2)

df['Close*'] = df['Close*'].str.replace(',', '')
df['Close*'] = df['Close*'].astype(float)

def calculate_returns(df_in):
    returns = (df_in[1] - df_in[0])/(df_in[0])*100
    return returns

df['returns'] = df[['Open', 'Close*']].apply(calculate_returns, axis = 1)

print(df.head())