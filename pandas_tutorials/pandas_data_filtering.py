#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:41:49 2020

@author: sadrachpierre
"""

import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


df = pd.read_csv('fifa_data.csv')
df['Wage'] = df['Wage'].str.lstrip('€')
df['Wage'] = df['Wage'].str.rstrip('K')
df['Wage'] = df['Wage'].astype(float)*1000.0
del df['Unnamed: 0']
print(df.head())

def filter_category(category, category_value):
    df_filter = df.loc[df[category] == category_value]
    return df_filter

df_filter = filter_category('Nationality', 'Argentina')
print(df_filter.head())


def filter_category_with_list(category, category_value_list):
    df_filter = df.loc[df[category].isin(category_value_list)]
    return df_filter


df_filter = filter_category_with_list('Nationality', ['Brazil', 'Spain', 'Argentina'])
print(df_filter.head())


df_filter = filter_category_with_list('Club', ['Manchester City', 'Real Madrid', 'FC Barcelona'])
print(df_filter.head())



def filter_numerical(numerical, numerical_value, relationship):
    if relationship == 'greater':
        df_filter = df.loc[df[numerical] > numerical_value]
    elif relationship == 'less':
        df_filter = df.loc[df[numerical] < numerical_value]     
    else: 
        df_filter = df.loc[df[numerical] == numerical_value]  
    return df_filter


df_filter = filter_numerical('Age', 30, 'greater')
print(df_filter.head())


df_filter = filter_numerical('Age', 30, 'less')
print(df_filter.head())


df_filter = filter_numerical('Age', 30, 'equal')
print(df_filter.head())


from collections import Counter

print(Counter(df['Wage']))


print(df['Wage'].head())

df_filter = filter_numerical('Wage', 100000, 'less')
print(df_filter.head())
