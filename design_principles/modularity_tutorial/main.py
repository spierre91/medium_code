#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 10:28:21 2023

@author: sadrach.pierre
"""
from data_loader import DataLoader
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator

if __name__ == '__main__':
    # Step 1: Read in the data
    data_loader = DataLoader('synthetic_transaction_data_Dining.csv')
    data = data_loader.read_data()

    # Step 2: Define the list of merchant states
    MERCHANT_STATES = ['New York', 'Florida']

    # Step 3: Create instances of the classes
    data_processor = DataProcessor()
    model_trainer = ModelTrainer()
    model_evaluator = ModelEvaluator()
    # Step 4: Iterate over merchant states and train models
    for state in MERCHANT_STATES:
        print("Evaluation for '{}' data:".format(state))

        filtered_data = data_processor.filter_data_by_state(data, state)

        X = filtered_data[['cardholder_name', 'card_number', 'card_type', 'merchant_name', 'merchant_category',
                           'merchant_state', 'merchant_city', 'transaction_amount', 'merchant_category_code']]
        y = filtered_data['fraud_flag']

        X_train, X_test, y_train, y_test = data_processor.split_data(X, y)

        model = model_trainer.train_model(X_train, y_train, cat_features=['cardholder_name', 'card_number',
                                                                           'card_type', 'merchant_name',
                                                                           'merchant_category', 'merchant_state',
                                                                           'merchant_city',
                                                                           'merchant_category_code'])

        precision, accuracy = model_evaluator.evaluate_model(model, X_test, y_test)
        print('Precision:', precision)
        print('Accuracy:', accuracy)
        print()