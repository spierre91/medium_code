#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:22:24 2023

@author: sadrach.pierre
"""

import numpy as np
import pandas as pd
import catboost as cb
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score

# Step 1: Read in the data
data = pd.read_csv('synthetic_transaction_data_Dining.csv')

# Step 2: Filter data frames for 'New York' and 'Florida'
ny_data = data[data['merchant_state'] == 'New York']
fl_data = data[data['merchant_state'] == 'Florida']

# Step 3: Split the data into training and testing sets
ny_X = ny_data[['cardholder_name', 'card_number', 'card_type', 'merchant_name', 'merchant_category',
                'merchant_state', 'merchant_city', 'transaction_amount', 'merchant_category_code']]
ny_y = ny_data['fraud_flag']
ny_X_train, ny_X_test, ny_y_train, ny_y_test = train_test_split(ny_X, ny_y, test_size=0.2, random_state=42)

fl_X = fl_data[['cardholder_name', 'card_number', 'card_type', 'merchant_name', 'merchant_category',
                'merchant_state', 'merchant_city', 'transaction_amount', 'merchant_category_code']]
fl_y = fl_data['fraud_flag']
fl_X_train, fl_X_test, fl_y_train, fl_y_test = train_test_split(fl_X, fl_y, test_size=0.2, random_state=42)

cats  = ['cardholder_name', 'card_number', 'card_type', 'merchant_name', 'merchant_category',
                'merchant_state', 'merchant_city', 'merchant_category_code']
# Step 4: Perform hyperparameter tuning with grid search and build models

# Hyperparameter tuning and model building for 'New York' data
param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [100, 200, 300]
    
}
ny_model = cb.CatBoostClassifier(cat_features = cats, random_state=42)
ny_grid_search = GridSearchCV(estimator=ny_model, param_grid=param_grid, cv=3)
ny_grid_search.fit(ny_X_train, ny_y_train)
ny_best_params = ny_grid_search.best_params_
ny_model = cb.CatBoostClassifier(cat_features = cats, random_state=42, **ny_best_params)
ny_model.fit(ny_X_train, ny_y_train)
ny_predictions = ny_model.predict(ny_X_test)
ny_precision = precision_score(ny_y_test, ny_predictions)
ny_accuracy = accuracy_score(ny_y_test, ny_predictions)

# Hyperparameter tuning and model building for 'Florida' data

fl_model = cb.CatBoostClassifier(cat_features = cats, random_state=42)
fl_grid_search = GridSearchCV(estimator=fl_model, param_grid=param_grid, cv=3)
fl_grid_search.fit(fl_X_train, fl_y_train)
fl_best_params = fl_grid_search.best_params_
fl_model = cb.CatBoostClassifier(cat_features = cats, random_state=42, **fl_best_params)
fl_model.fit(fl_X_train, fl_y_train)
fl_predictions = fl_model.predict(fl_X_test)
fl_precision = precision_score(fl_y_test, fl_predictions)
fl_accuracy = accuracy_score(fl_y_test, fl_predictions)

# Step 5: Evaluate the models
print("Evaluation for 'New York' data:")
print('Precision:', ny_precision)
print('Accuracy:', ny_accuracy)
print("Evaluation for 'Florida' data:")
print('Precision:', fl_precision)
print('Accuracy:', fl_accuracy)
