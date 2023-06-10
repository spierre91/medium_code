#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 10:27:28 2023

@author: sadrach.pierre
"""

from sklearn.metrics import precision_score, accuracy_score

class ModelEvaluator:
    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        precision = precision_score(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)
        return precision, accuracy