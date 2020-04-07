import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("hotel_bookings.csv")

print(df.head())

condition = df['market_segment'] == 'Direct'
print(condition.head())

print(df[condition].head())

from collections import Counter

print(Counter(df['arrival_date_month']))

condition2 = df['arrival_date_month'] == 'December'
print(df[condition2].head())

print(Counter(df['stays_in_week_nights']))

condition3 = df['stays_in_week_nights'] >= 5
print(df[condition3].head())

print(df.loc[condition3,  'arrival_date_month'].head())

condition3 = (df['stays_in_week_nights'] >= 5)
condition4 = (df['arrival_date_month'] == 'August')
print(df.loc[condition3 & condition4].head())

condition3 = (df['stays_in_week_nights'] >= 5)
condition4 = (df['arrival_date_month'] == 'August')
print(df.loc[condition3 | condition4].head())
