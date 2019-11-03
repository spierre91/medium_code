#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 20:37:23 2019

@author: sadrachpierre
"""

import yfinance as yf
import seaborn as sns
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = yf.Ticker('BYND')
df = data.history(period="max",  start="2019-05-01", end="2019-11-02")
print(df.head())



sns.set()
df['timestamp'] = df.index
df['timestamp'] = pd.to_datetime(df['timestamp'])
sns.lineplot(df['timestamp'], df['Open'])
plt.ylabel("Open Price")


df['returns'] = (df['Close']-df['Open'])/df['Open']
sns.lineplot(df['timestamp'], df['returns'])
plt.ylabel("Returns")


forecast_out = 3
df['prediction'] = df[['Close']].shift(-forecast_out)
X = np.array(df['Close']).reshape(-1,1)
X = X[:-forecast_out]
y = np.array(df['prediction'])
y = y[:-forecast_out]


reg = RandomForestRegressor(n_estimators = 300, max_depth =300, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 7)
reg.fit(X_train, y_train)
print("Performance (R^2): ", reg.score(X_test, y_test))