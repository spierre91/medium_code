#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:10:26 2020

@author: sadrachpierre
"""

import pandas as pd
df = pd.read_csv("superbowl.csv")
print(df.columns)

print(df.head())
def return_counter(data_frame, column_name, limit):
   from collections import Counter    
   print(dict(Counter(data_frame[column_name].values).most_common(limit)))
return_counter(df, 'MVP', 5)
return_counter(df, 'Stadium', 5)
return_counter(df, 'Winner', 5)


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


stats = return_statistics(df, 'Winner', 'Winner Pts')
print(stats.head(15))


def get_boxplot_of_categories(data_frame, categorical_column, numerical_column, limit):
    import seaborn as sns
    from collections import Counter
    keys = []
    for i in dict(Counter(df[categorical_column].values).most_common(limit)):
        keys.append(i)
    print(keys)
    
    df_new = df[df[categorical_column].isin(keys)]
    sns.set()
    sns.boxplot(x = df_new[categorical_column], y = df_new[numerical_column])
#get_boxplot_of_categories(df, 'Winner', 'Winner Pts', 5)

def get_histogram(data_frame, numerical_column):
    df_new = data_frame
    df_new[numerical_column].hist(bins=100)

#get_histogram(df, 'Winner Pts')    
    
df['Date'] = pd.to_datetime(df['Date'])

def get_time_series(data_frame, numerical_column):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    df_new = data_frame
    plt.plot(df_new['Date'], df_new[numerical_column])
    plt.xlabel('Date')
    plt.ylabel(numerical_column)

print(df.head())

get_time_series(df, 'Winner Pts')