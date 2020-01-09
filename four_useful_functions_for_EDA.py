#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:47:13 2020

@author: sadrachpierre
"""

import pandas as pd 
from collections import Counter 
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('winemag-data_first150k.csv')
print(df.head())


def return_counter(data_frame, column_name, limit):
       
    print(dict(Counter(data_frame[column_name].values).most_common(limit)))
    
return_counter(df, 'variety', 5)




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

stats = return_statistics(df, 'variety', 'price')
print(stats.head())


stats = return_statistics(df, 'country', 'price')
print(stats.head())



def get_boxplot_of_categories(data_frame, categorical_column, numerical_column, limit):
    import seaborn as sns
    import matplotlib.pyplot as plt
    keys = []
    for i in dict(Counter(df[categorical_column].values).most_common(limit)):
        keys.append(i)
    print(keys)
    
    df_new = df[df[categorical_column].isin(keys)]
    sns.set()
    sns.boxplot(x = df_new[categorical_column], y = df_new[numerical_column])
    plt.show()
    
get_boxplot_of_categories(df, 'variety', 'price', 3)



def scatter_plot_category(data_frame, categorical_column, categorical_value, numerical_column_one, numerical_column_two):
    df_new = data_frame[data_frame[categorical_column] == categorical_value]
    sns.set()
    plt.scatter(x= df_new[numerical_column_one], y = df_new[numerical_column_two])
    plt.xlabel(numerical_column_one)
    plt.ylabel(numerical_column_two)
    
    
    
scatter_plot_category(df, 'country', 'US', 'points', 'price')
    