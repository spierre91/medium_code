#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:07:04 2022

@author: sadrachpierre
"""
import pandas as pd 

df = pd.read_csv("uci_bc_data.csv")

print(df.head()) #default displays first 5 rows
print(df.head(3)) #display first 3 rows
print(df.head(10)) # display first 10 rows


print(df.tail()) #display last 5 rows by default
print(df.tail(10)) #display last 10 rows

print(df.columns)
df_selected = df[['id', 'diagnosis', 'radius_mean']].copy() #select columns

print(df_selected.head())


df_malignant = df[df['diagnosis']=='M'].copy() # filter on malignant tumors
df_benign= df[df['diagnosis']=='B'].copy() #filter on benign tumors 
 




mean_raidus = df['radius_mean'].mean()
print("Mean Radius: ", mean_raidus)

df_above_average = df[df['radius_mean'] > mean_raidus].copy()

df_grouped = df.groupby(['diagnosis'])['radius_mean'].mean()
print(df_grouped)
