#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:50:50 2020

@author: sadrachpierre
"""

import pandas as pd 

df = pd.read_csv("QHP_landscape_2020.csv")

print(list(df.columns))

df = df[['Issuer Name', 'County Name', 'Plan Marketing Name',   
         'Medical Maximum Out Of Pocket - Individual - Standard',
         'Medical Deductible - Individual - Standard', 'Primary Care Physician - Standard',
         'Specialist - Standard', 'Emergency Room - Standard']]
         
print(df.head())


def return_counter(data_frame, column_name, limit):
   from collections import Counter    
   print(dict(Counter(data_frame[column_name].values).most_common(limit)))
   
return_counter(df, 'Issuer Name', 5)
   
df['Weight_kg'] = df['Medical Maximum Out Of Pocket - Individual - Standard'].str[1:].astype(float)/2.20462
import numpy as np 
OOP_INDV = []
for i in list(df['Medical Maximum Out Of Pocket - Individual - Standard'].values):
    try:
        OOP_INDV.append(float((str(i)[1] + str(i)[3:])))
    except(ValueError):
        OOP_INDV.append(np.nan)
        
df['OOP_INDV'] = OOP_INDV
print(df[['Medical Maximum Out Of Pocket - Individual - Standard', 'OOP_INDV']].head())

DEDUCT_INDV = []
for i in list(df['Medical Deductible - Individual - Standard'].values):
    try:
        DEDUCT_INDV.append(float((str(i)[1] + str(i)[3:])))
    except(ValueError):
        DEDUCT_INDV.append(np.nan)
        
df['DEDUCT_INDV'] = DEDUCT_INDV
print(df[['Medical Deductible - Individual - Standard', 'DEDUCT_INDV']].head())
def return_statistics(data_frame, categorical_column, numerical_column):
    mean = []
    std = []
    field = []
    for i in set(list(data_frame[categorical_column].values)):
        new_data = data_frame[data_frame[categorical_column] == i]
        field.append(i)
        mean.append(new_data[numerical_column].mean())
        std.append(new_data[numerical_column].std())
    df = pd.DataFrame({'{}'.format(categorical_column): field, 'mean {}'.format(numerical_column): mean, 'std in {}'.format(numerical_column): std})
    df.sort_values('mean {}'.format(numerical_column), inplace = True, ascending = False)
    df.dropna(inplace = True)
    return df
stats = return_statistics(df, 'County Name', 'OOP_INDV')
print(stats.head(15))

stats = return_statistics(df, 'Issuer Name', 'DEDUCT_INDV')
print(stats.head(15))
return_counter(df, 'County Name', 5)
def get_boxplot_of_categories(data_frame, categorical_column, numerical_column, limit):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import Counter
    keys = []
    for i in dict(Counter(df[categorical_column].values).most_common(limit)):
        keys.append(i)
    print(keys)
    
    df_new = df[df[categorical_column].isin(keys)]
    sns.boxplot(x = df_new[categorical_column], y = df_new[numerical_column])
    
def get_scatter_plot_category(data_frame, categorical_column, categorical_value, numerical_column_one, numerical_column_two):
    import matplotlib.pyplot as plt
    import seaborn as sns
    df_new = data_frame[data_frame[categorical_column] == categorical_value]
    sns.set()
    plt.scatter(x= df_new[numerical_column_one], y = df_new[numerical_column_two])
    plt.xlabel(numerical_column_one)
    plt.ylabel(numerical_column_two)
    
get_scatter_plot_category(df, 'Issuer Name', 'Medica', 'DEDUCT_INDV', 'OOP_INDV')
get_boxplot_of_categories(df, 'Issuer Name', 'DEDUCT_INDV', 5)

get_boxplot_of_categories(df, 'Issuer Name', 'OOP_INDV', 4)