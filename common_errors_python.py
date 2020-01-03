#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:45:39 2020

@author: sadrachpierre
"""

import pandas as pd
import seaborn as sns
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 
sns.set()
df = pd.read_csv("weight-height.csv")

print(df.head())

plt.scatter(df['Weight'],  df['Height'])
plt.xlabel("Weight")
plt.ylabel("Height")
plt.show()


X = np.array(df["Weight"]).reshape(-1,1)
y = np.array(df["Height"]).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.33)


reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("R^2 Accuracy: ", reg.score(X_test, y_test))