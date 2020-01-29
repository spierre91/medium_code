#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:18:25 2020

@author: sadrachpierre
"""



import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



df = pd.read_csv("housing.csv")
print(df.head())


import numpy as np
X = np.array(df[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']])
y = np.array(df['median_house_value'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(16, input_shape = (8,)))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))
model.compile(optimizer='adam', loss='mse', metrics =['mape'])
model.fit(X_train, y_train, epochs = 500, batch_size = 100)

y_pred = model.predict(X_test)
print(y_pred)
print(y_test)
import matplotlib.pyplot as plt

plt.clf()
fig = plt.figure()
fig.suptitle('Scatter plot of Actual versus Predicted')
plt.scatter(x=y_pred, y=y_test, marker='.')
plt.xlabel('Predicted')
plt.ylabel('Actual ')
plt.show()



