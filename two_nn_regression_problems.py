#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:48:54 2020

@author: sadrachpierre
"""

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("cars.csv")
print(df.head())
X = np.array(df[['age', 'gender', 'miles', 'debt', 'income']])
y = np.array(df['sales'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

y_train=np.reshape(y_train, (-1,1))

X_scaled = MinMaxScaler()
y_scaled = MinMaxScaler()
X_test_scaled = MinMaxScaler()


X_scaled.fit(X_train)
X_scaled = X_scaled.transform(X_train)

#X_test_scaled.fit(X_test)
#X_test_scaled = X_test_scaled.transform(X_test)


y_scaled.fit(y_train)
y_scaled = y_scaled.transform(y_train)





#print(df.head())
#
model = Sequential()
model.add(Dense(64, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'], validation_split = 0.2)
model.fit(X_scaled, y_scaled, epochs=100, batch_size=10 )

y_pred = model.predict(X_test)

import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')

#column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
#                'Acceleration', 'Model Year', 'Origin']
#
##df = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
##df = pd.read_csv(df, names=column_names,
##                      na_values = "?", comment='\t',
##                      sep=" ", skipinitialspace=True)
#df = pd.read_csv("auto-mpg.csv")
#df.dropna(inplace = True)
#
#print(df.head())
#X = np.array(df[['Cylinders','Displacement','Horsepower','Weight',
#                'Acceleration', 'Model Year', 'Origin']])
#y = np.array(df['MPG'])
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
#
#y_train=np.reshape(y_train, (-1,1))
#scaler_x = MinMaxScaler()
##scaler_y = MinMaxScaler()
#scaler_x_t = MinMaxScaler()
#
##print(scaler_y.fit(y_train))
#print(scaler_x.fit(X_train))
#print(scaler_x_t.fit(X_test))
#xscale=scaler_x.transform(X_train)
#xtscale=scaler_x_t.transform(X_test)
##yscale=scaler_y.transform(y_train)
#
#
#
#model = Sequential()
#model.add(Dense(64, input_dim=7, kernel_initializer='normal', activation='relu'))
#model.add(Dense(64, activation='relu'))
#model.add(Dense(1, activation='linear'))
#optimizer = keras.optimizers.RMSprop(0.001)
#model.compile(loss='mse', optimizer=optimizer, metrics=['mse','mae'], validation_split = 0.2)
#model.fit(xscale, y_train, epochs=1000, batch_size=50, )
#y_pred = model.predict(xtscale)
#
#import matplotlib.pyplot as plt 
#plt.scatter(y_test, y_pred)
#plt.xlabel('True Values [MPG]')
#plt.ylabel('Predictions [MPG]')
