#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 10:24:21 2023

@author: sadrach.pierre
"""

import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_data(self):
        return pd.read_csv(self.file_path)