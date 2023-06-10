#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 10:26:11 2023

@author: sadrach.pierre
"""

from sklearn.model_selection import train_test_split

class DataProcessor:
    def filter_data_by_state(self, data, state):
        return data[data['merchant_state'] == state]

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)