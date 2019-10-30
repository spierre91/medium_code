#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:21:38 2019

@author: sadrachpierre
"""

from pytrends.request import TrendReq
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

def get_sum(key_words):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([key_words], cat=0, timeframe='2019-10-01 2019-10-30',  gprop='',geo="US-NY")    
    df = pytrends.interest_over_time()  
    return df[key_words].sum()


def get_data(key_words):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([key_words], cat=0, timeframe='2019-10-01 2019-10-30',  gprop='',geo="US-NY")    
    df = pytrends.interest_over_time()  
    print(df.head(10))
    df['timestamp'] = df.index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def plot_results(df, key_words):
    sns.set()
    ax = sns.lineplot(df['timestamp'], df[key_words], label = key_words)
    plt.ylabel("Number of Searches")
    ax.legend()
    plt.show()



results = pd.DataFrame({"Kim Kardashian Costume":[get_sum("Kim Kardashian Costume")],"Taylor Swift Costume":[get_sum("Taylor Swift Costume")],
           "Trump Costume": [get_sum("Trump Costume")], "Joker Costume": [get_sum("Joker Costume")]})

print(results)