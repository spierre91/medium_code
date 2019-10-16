#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:11:09 2019

@author: sadrachpierre
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def initialize_data(state):
    df = pd.read_csv("amazon.csv",  encoding = "ISO-8859-1")
    df['month_cat'] = df['month'].astype('category')
    df['month_cat'] = df['month_cat'].cat.codes    
    df = df[df['state'] == state]
    return df

def train_test_split(year, df):    
    df_train = df[df['year'] < year]
    df_test = df[df['year'] == year]
    X_train  = np.array(df_train[['year', 'month_cat']])
    y_train  = np.array(df_train['number'])
    X_test  = np.array(df_test[['year', 'month_cat']])
    y_test  = np.array(df_test['number'])    
    return X_train, X_test, y_train, y_test

def model_tuning(N_ESTIMATORS, MAX_DEPTH):
    model = RandomForestRegressor(n_estimators = N_ESTIMATORS, max_depth = MAX_DEPTH, random_state = 42)
    return model 

def predict_fire(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).astype(int)
    mae = mean_absolute_error(y_pred, y_test)
    print("Mean Absolute Error: ", mae)   
    df_results = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
    print(df_results.head())
    
def main():
    df = pd.read_csv("amazon.csv",  encoding = "ISO-8859-1")
    for i in list(set(df['state'].values)):   
        df = initialize_data(i)
        X_train, X_test, y_train, y_test = train_test_split(2017, df)
        model = model_tuning(50, 50)
        predict_fire(model, X_train, X_test, y_train, y_test)  
        print(i)
            
if __name__ == "__main__":     
    main()
    
    
'''
def main():
    df = initialize_data('Sergipe')
    X_train, X_test, y_train, y_test = train_test_split(2017, df)
    model = model_tuning(50, 50)
    predict_fire(model, X_train, X_test, y_train, y_test)
    print('Sergipe')
    
    
    df = initialize_data('Distrito Federal')
    X_train, X_test, y_train, y_test = train_test_split(2017, df)
    model = model_tuning(50, 50)
    predict_fire(model, X_train, X_test, y_train, y_test)
    print('Distrito Federal')
'''
