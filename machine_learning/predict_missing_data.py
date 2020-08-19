#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:40:39 2020

@author: sadrachpierre
"""
import pandas as pd 
import numpy as np 
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
df = pd.read_csv("winemag-data-130k-v2.csv").sample(n=5000, random_state = 42)


del df['Unnamed: 0']
print(df.head())
print(df.info())
df['country_cat'] = df['country'].astype('category')
df['country_cat'] = df['country_cat'].cat.codes

df['province_cat'] = df['province'].astype('category')
df['province_cat'] = df['province_cat'].cat.codes

df['winery_cat'] = df['winery'].astype('category')
df['winery_cat'] = df['winery_cat'].cat.codes

df['variety_cat'] = df['variety'].astype('category')
df['variety_cat'] = df['variety_cat'].cat.codes

df_filter = df[df['price'] > 0].copy()
df_filter = df_filter[df_filter['price'] <= df_filter['price'].mean() + df_filter['price'].std() ].copy()

from sklearn.linear_model import LinearRegression

print("Correlation: ", df['points'].corr(df['price']))

kf = KFold(n_splits=10, random_state = 42)

y_pred = []
y_true = []

for train_index, test_index in kf.split(df_filter):
    df_test = df_filter.iloc[test_index]
    df_train = df_filter.iloc[train_index]
    
    X_train = np.array(df_train['points']).reshape(-1, 1)     
    y_train = np.array(df_train['price']).reshape(-1, 1)
    X_test = np.array(df_test['points']).reshape(-1, 1)  
    y_test = np.array(df_test['price']).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred.append(model.predict(X_test)[0])
    y_true.append(y_test[0])    
    


from sklearn.ensemble import RandomForestRegressor

kf = KFold(n_splits=10, random_state = 42)

y_pred_rf = []
y_true_rf = []

features = ['points', 'country_cat', 'province_cat', 'winery_cat', 'variety_cat']

for train_index, test_index in kf.split(df_filter):
    df_test = df_filter.iloc[test_index]
    df_train = df_filter.iloc[train_index]
    
    X_train = np.array(df_train[features])
    y_train = np.array(df_train['price'])
    X_test = np.array(df_test[features])
    y_test = np.array(df_test['price'])
    model = RandomForestRegressor(n_estimators = 1000, max_depth = 1000, random_state = 42)
    model.fit(X_train, y_train)

    y_pred_rf.append(model.predict(X_test)[0])
    y_true_rf.append(y_test[0])    

print("Mean Square Error (Random Forest): ", mean_squared_error(y_pred_rf, y_true_rf))
print("Mean Square Error (Linear Regression): ", mean_squared_error(y_true, y_pred))