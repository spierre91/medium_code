#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:22:18 2023

@author: sadrach.pierre
"""

import numpy as np
import pandas as pd
import catboost as cb
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
import matplotlib.pyplot as plt
# Step 1: Read in the data
data = pd.read_csv('synthetic_transaction_data_Dining.csv')

# Step 2: Define the list of merchant states
MERCHANT_STATES = ['New York', 'Florida']

# Step 3: Iterate over merchant states, perform model training and evaluation
cats  = ['cardholder_name', 'card_number', 'card_type', 'merchant_name', 'merchant_category',
                'merchant_state', 'merchant_city', 'merchant_category_code']
for state in MERCHANT_STATES:
    print("Evaluation for '{}' data:".format(state))
    
    # Filter data frames for the current state
    filtered_data = data[data['merchant_state'] == state]
    
    # Split the data into training and testing sets
    X = filtered_data[['cardholder_name', 'card_number', 'card_type', 'merchant_name', 'merchant_category',
                       'merchant_state', 'merchant_city', 'transaction_amount', 'merchant_category_code']]
    y = filtered_data['fraud_flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Perform hyperparameter tuning with grid search and build the model
    param_grid = {
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [100, 200, 300]
    }
    model = cb.CatBoostClassifier(cat_features=cats, random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    model = cb.CatBoostClassifier(cat_features=X.columns, random_state=42, **best_params)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    precision = precision_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    print('Precision:', precision)
    print('Accuracy:', accuracy)
    
    # Visualize the confusion matrix
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    print('\n')
