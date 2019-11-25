#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:37:42 2019

@author: sadrachpierre
"""

import pandas as pd

print("READING DATA")
df = pd.read_csv("aac_shelter_outcomes.csv")
print(df.head())
print(df.tail())
print(df.columns)


print("CLEANING DATA")
df = pd.read_csv("aac_shelter_outcomes.csv")
print(df.isnull().sum())
print("Length Before:", len(df))
df.dropna(inplace=True)
print("Length After:", len(df))

df = pd.read_csv("aac_shelter_outcomes.csv")
print("Length Before:", len(df))
df.fillna(0, inplace=True)
print("Length After:", len(df))

print("FILTERING DATA")
df = df[df['breed'== 'Mastiff Mix']]
df = df[df['sex_upon_outcome'] == 'Spayed Female']
print(df.head())

df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])

df['week_of_birth'] = df['date_of_birth'].dt.week
df['month_of_birth'] = df['date_of_birth'].dt.month
df['year_of_birth'] = df['date_of_birth'].dt.year
print(df.head())

df = df[df['month_of_birth'] == 1]
print(df.head())

df = pd.read_csv("aac_shelter_outcomes.csv")

df = df[df['month_of_birth'] == 1].reset_index()
print(df.head())

df = df.set_index('month_of_birth')
print(df.head())


print("SELECTING ROWS AND COLUMNS")

print(df.head())
print("---------------------First---------------------")
print(df.iloc[0])
print("---------------------Second---------------------") 
print(df.iloc[1])
print("---------------------Last---------------------")
print(df.iloc[-1])

print("---------------------First---------------------")
print(df.loc[0, 'breed'])
print("---------------------Second---------------------") 
print(df.loc[1, 'breed'])


print("---------------------First---------------------")
print(df.loc[0:3, 'breed'])
print("---------------------Second---------------------") 
print(df.loc[3:6, 'breed'])


print("AGGREGRATING DATA")

df = pd.read_csv("aac_shelter_outcomes.csv")
df  = df.groupby('year_of_birth')['breed'].count()
print(df.head())


df = pd.read_csv("aac_shelter_outcomes.csv")
df  = df.groupby('breed')['week_of_birth'].mean()
print(df.head())

print("WRITING TO A FILE")
df.to_csv("new_name_of_file.csv")