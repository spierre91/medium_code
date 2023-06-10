#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 10:26:33 2023

@author: sadrach.pierre
"""

import catboost as cb
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def train_model(self, X_train, y_train, cat_features):
        param_grid = {
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [100, 200, 300]
        }
        model = cb.CatBoostClassifier(cat_features=cat_features, random_state=42, verbose=0)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        model = cb.CatBoostClassifier(cat_features=cat_features, random_state=42, **best_params, verbose=0)
        model.fit(X_train, y_train)
        return model