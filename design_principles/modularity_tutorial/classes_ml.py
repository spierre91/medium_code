#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 09:51:38 2023

@author: sadrach.pierre
"""

import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, accuracy_score

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_data(self):
        return pd.read_csv(self.file_path)

    def filter_data_by_state(self, data, state):
        return data[data['merchant_state'] == state]

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)


class ModelTrainer:
    def __init__(self, cat_features):
        self.cat_features = cat_features

    def train_model(self, X_train, y_train):
        param_grid = {
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [100, 200, 300]
        }
         
        model = cb.CatBoostClassifier(cat_features=self.cat_features, random_state=42, verbose=0)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        model = cb.CatBoostClassifier(cat_features=self.cat_features, random_state=42, **best_params, verbose=0)
        model.fit(X_train, y_train)
        return model


class ModelEvaluator:
    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        precision = precision_score(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)
        return precision, accuracy


class MachineLearningApp:
    def __init__(self, file_path, cat_features):
        self.data_processor = DataProcessor(file_path)
        self.model_trainer = ModelTrainer(cat_features)
        self.model_evaluator = ModelEvaluator()

    def run(self):
        # Step 1: Read in the data
        data = self.data_processor.read_data()

        # Step 2: Define the list of merchant states
        MERCHANT_STATES = ['New York', 'Florida']

        # Step 3: Iterate over merchant states and train models
        for state in MERCHANT_STATES:
            print("Evaluation for '{}' data:".format(state))

            filtered_data = self.data_processor.filter_data_by_state(data, state)

            X = filtered_data[['cardholder_name', 'card_number', 'card_type', 'merchant_name', 'merchant_category',
                               'merchant_state', 'merchant_city', 'transaction_amount', 'merchant_category_code']]
            y = filtered_data['fraud_flag']

            X_train, X_test, y_train, y_test = self.data_processor.split_data(X, y)

            model = self.model_trainer.train_model(X_train, y_train)

            precision, accuracy = self.model_evaluator.evaluate_model(model, X_test, y_test)
            print('Precision:', precision)
            print('Accuracy:', accuracy)
            print('\n')


if __name__ == '__main__':
    app = MachineLearningApp('synthetic_transaction_data_Dining.csv', ['cardholder_name', 'card_number',
                                                                        'card_type', 'merchant_name',
                                                                        'merchant_category', 'merchant_state',
                                                                        'merchant_city',
                                                                        'merchant_category_code'])
    app.run()
