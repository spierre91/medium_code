#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:48:47 2019

@author: sadrachpierre
"""

import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt


df_global_temp = pd.read_csv("annual_temp.csv")


print(df_global_temp.head())

df_global_temp = df_global_temp[df_global_temp['Source'] == 'GISTEMP'].reset_index()[["Source", "Year", "Mean"]]

print(df_global_temp.head())




df_rice = pd.read_csv("rice-yields.csv")
df_rice = df_rice[df_rice['Entity']=='Nigeria'].reset_index()[["Entity", "Code", "Year", ' (tonnes per hectare)']]
print(df_rice.head())
sns.set()
sns.lineplot(df_global_temp['Year'], df_global_temp['Mean'])
plt.ylabel("Mean")
plt.title("Average Global Mean Temperature and wheat production in Nigeria")

sns.lineplot(df_rice['Year'], df_rice[' (tonnes per hectare)'])
plt.ylabel("Mean temperature/tonnes per hectare wheat in Nigeria ")
