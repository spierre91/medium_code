#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:08:24 2020

@author: sadrachpierre
"""

import pandas as pd 
df = pd.read_csv("weatherHistory.csv")

df['Formatted Date'] = pd.to_datetime(df['Formatted Date'],  utc=True,)
df['year'] = df['Formatted Date'].dt.year
df['month'] = df['Formatted Date'].dt.month
df['day'] = df['Formatted Date'].dt.day



import numpy as np 

X = np.array(df[[ 'Humidity', 'year', 'month', 'day', 'Pressure (millibars)', 'Visibility (km)', 'Wind Bearing (degrees)']])
y = np.array(df['Temperature (C)'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)
y_pred = reg_rf.predict(X_test)
from sklearn import metrics
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

feature_df = pd.DataFrame({'Importance':reg_rf.feature_importances_, 'Features': [ 'Humidity', 'year', 'month', 'day', 'Pressure (millibars)', 'Visibility (km)', 'Wind Bearing (degrees)'] })
print(feature_df)

from sklearn.svm import SVR

reg_svr = SVR()
reg_svr.fit(X_train, y_train)
y_pred = reg_svr.predict(X_test)
from sklearn import metrics
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))


from sklearn.neighbors import KNeighborsRegressor

reg_knn = KNeighborsRegressor()
reg_knn.fit(X_train, y_train)
y_pred = reg_knn.predict(X_test)
from sklearn import metrics
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))