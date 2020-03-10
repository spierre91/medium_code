#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:32:08 2020

@author: sadrachpierre
"""
from pytrends.request import TrendReq

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 


def get_searches(key_word, state):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([key_word], cat=0, timeframe='2020-02-01 2020-03-10',  gprop='',geo='US-{}'.format(state))    
    df = pytrends.interest_over_time()
            
    
    print(df.head())
    
    sns.set()
    df['timestamp'] = pd.to_datetime(df.index)
    sns.lineplot(df['timestamp'], df[key_word])
    
    plt.title("Normalized Searches for Coronavirus in NY (blue), MA (orange), and CA (green)".format(key_word, state))
    plt.ylabel("Number of Searches")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    
get_searches('Coronavirus', 'NY')
get_searches('Coronavirus', 'MA')
get_searches('Coronavirus', 'CA')