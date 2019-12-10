#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:20:34 2019

@author: sadrachpierre
"""

import pandas as pd 
import numpy as np 


pd.set_option('display.max_columns', 1000000)


df = pd.read_csv("online_retail.csv")

np.random.seed(32)
df['column_with_bad_values'] = np.random.randint(4, size = len(df))
df['column_with_bad_values'].iloc[0] = np.nan
df['column_with_bad_values'].iloc[2] = np.nan
df['column_with_bad_values'].iloc[4] = np.nan
print(df.head())


df['column_with_bad_values'].fillna(df['column_with_bad_values'].mean(), inplace = True)
print(df.head())



import pandas as pd 

#read in data 
df = pd.read_csv("online_retail.csv")
print(df.head())


#Impute Missing/Bad Numerical Values with Zero
np.random.seed(42)
df['column_with_bad_values'] = df['UnitPrice']/np.random.randint(4, size = len(df))
df['column_with_bad_values'].iloc[3] = np.nan
print(df.head())

df['column_with_bad_values'].fillna(0, inplace = True)
print(df.head())

df.loc[np.isinf(df['column_with_bad_values']), 'column_with_bad_values'] = 0
print(df.head())

#Impute Missing Numerical Values with Mean
df = pd.read_csv("online_retail.csv")
np.random.seed(42)
df['column_with_bad_values'] = np.random.randint(4, size = len(df))
df['column_with_bad_values'].iloc[0] = np.nan
df['column_with_bad_values'].iloc[2] = np.nan
df['column_with_bad_values'].iloc[4] = np.nan
print(df.head())

df['column_with_bad_values'].fillna(df['column_with_bad_values'].mean(), inplace = True)
print(df.head())

#impute infinite and 'nan' with mean
np.random.seed(32)
df['column_with_bad_values'] = df['UnitPrice']/np.random.randint(4, size= len(df))
df['column_with_bad_values'].iloc[0] = np.nan
df['column_with_bad_values'].iloc[2] = np.nan
print(df.head())

df.loc[np.isinf(df['column_with_bad_values']), 'column_with_bad_values'] = np.nan
df['column_with_bad_values'].fillna(df['column_with_bad_values'].mean(), inplace = True)
print(df.head())


#Impute Missing/Bad Numerical Values with Random Numbers from Normal Distribution
df = pd.read_csv("online_retail.csv")
np.random.seed(32)
df['column_with_bad_values'] = np.random.randint(4, size= len(df))
df['column_with_bad_values'].iloc[0] = np.nan
df['column_with_bad_values'].iloc[2] = np.nan
print(df.head())

mu, sigma = df['column_with_bad_values'].mean(), df['column_with_bad_values'].std()
df['column_with_bad_values'].fillna(np.random.normal(mu, sigma), inplace = True)
print(df.head())


#Impute Missing/Bad Categorical Variables with Mode
df = pd.read_csv("online_retail.csv")
df['column_with_bad_values'] = df['Description']
df['column_with_bad_values'].iloc[0] = np.nan
df['column_with_bad_values'].iloc[2] = np.nan
print(df.head())


from statistics import mode
mode = mode(list(df['column_with_bad_values'].values))
df['column_with_bad_values'].fillna(mode, inplace = True)
print(df.head())

#Log Transform of Numerical Data
df = pd.read_csv("online_retail.csv")

df['UnitPrice'] = df['UnitPrice'].astype(int)
df = df[df['UnitPrice'] >= 5]
df = df[df['UnitPrice'] <= 30]
print(df.head())
df['UnitPrice'].hist()

df['log_price'] = np.log(df['UnitPrice'])
df['log_price'].hist()

#Feature Engineering using np.where()
df = pd.read_csv("online_retail.csv")
print(df.head())

df['bool_description'] = np.where(df['Description'] == 'WHITE HANGING HEART T-LIGHT HOLDER', True, False)
print(df.head())

description_list = ['WHITE HANGING HEART T-LIGHT HOLDER',
'CREAM CUPID HEARTS COAT HANGER', 'RED WOOLLY HOTTIE WHITE HEART.']
df['bool_description'] = np.where(df['Description'].isin(description_list), True, False)
print(df.head())

df['string_description'] = np.where(df['Description'].isin(description_list), 'Yes', 'No')
print(df.head())

df['int_description'] = np.where(df['Description'].isin(description_list), 1, 0)
print(df.head())

#DATA LABELING
df = pd.read_csv("online_retail.csv")
df['new_target'] = np.where(df['Quantity'] >= 10, 1, 0)
print(df.head())

