#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 09:05:34 2019

@author: sadrachpierre
"""
import numpy as np 
import pandas as pd
df = pd.read_csv("data.csv")
print(df.head())

df = df[['Name', 'Age', 'Nationality', 'Value', 'Wage', 'Preferred Foot', 'Height', 'Weight', 'Position', 'Overall']]
print(df.head())

print(df['Height'].head())


Height_cm = []

for i in list(df['Height'].values):
    try:
        Height_cm.append((float(str(i)[0])*12.0 + float(str(i)[2:]))*2.54)
    except(ValueError):
        Height_cm.append(np.nan)
print(len(df))        
df['Height_cm'] = Height_cm
print(df['Height_cm'].head())

print("Mean Height (cm): ", df['Height_cm'].mean())
print("Standard Deviation in Height (cm): ", df['Height_cm'].std())


import seaborn as sns 
import matplotlib.pyplot as plt 
sns.set()
plt.title("Height Histogram")
df['Height_cm'].hist(bins = 10)


df.dropna(inplace = True)



df['Weight_kg'] = df['Weight'].str[:3].astype(float)/2.20462


print(df.head())


import matplotlib.pyplot as plt

def get_statistics(numeric_column_name):
    print("Mean {}: ".format(numeric_column_name), df[numeric_column_name].mean())
    print("Standard Deviation in {}: ".format(numeric_column_name), df[numeric_column_name].std())
    
    
    import seaborn as sns 
    import matplotlib.pyplot as plt 
    sns.set()
    plt.title("{} Histogram".format(numeric_column_name))
    df[numeric_column_name].hist(bins = 10)
    
get_statistics('Weight_kg')

from collections import Counter
print(dict(Counter(df['Nationality'].values).most_common(10)))


bar_plot = dict(Counter(df['Nationality'].values).most_common(5))
plt.bar(*zip(*bar_plot.items()))
plt.show()

def plot_most_common(category):
    bar_plot = dict(Counter(df[category].values).most_common(5))
    plt.bar(*zip(*bar_plot.items()))
    plt.show()
    
plot_most_common('Position')

df['Age'] = df['Age'].astype(int)
df['Wage'] = df['Wage'].str[1:]
df['Wage'] = df['Wage'].str[:-1]

df['Value'] = df['Value'].str[:-1]
df['Value'] = df['Value'].str[1:]

wage_list = []
value_list = []
for i in list(df['Wage'].values):
    try:
        wage_list.append(float(i)*1e3)
    except(ValueError):
        wage_list.append(np.nan)

for i in list(df['Value'].values):
    try:
        value_list.append(float(i)*1e6)
    except(ValueError):
        value_list.append(np.nan)

df['Wage_numeric'] = wage_list
df['Value_numeric'] = value_list
numerical_columns = df[['Height_cm', 'Weight_kg', 'Value_numeric', 'Age', 'Wage_numeric']]

print(numerical_columns.head())
import seaborn as sns
sns.set()
sns.heatmap(numerical_columns.corr(), annot=True)
plt.show()
df = df[df['Nationality'].isin(['England', 'Germany', 'Spain'])]
sns.boxplot(x= df['Nationality'], y = df['Height_cm'])
plt.show()


