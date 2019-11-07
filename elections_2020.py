#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:00:27 2019

@author: sadrachpierre
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from pytrends.request import TrendReq

def get_searches(candidate, state):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([candidate], cat=0, timeframe='2019-01-01 2019-10-31',  gprop='',geo='US-{}'.format(state))    
    df = pytrends.interest_over_time()
            
    
    print(df.head())
    
    sns.set()
    df['timestamp'] = pd.to_datetime(df.index)
    sns.lineplot(df['timestamp'], df[candidate])
    
    plt.title("Normalized Searches for Biden, Warren and Sanders in {}".format(state))
    plt.ylabel("Number of Searches")
    plt.xlabel("Date")
    

get_searches('Biden', 'MA')
get_searches('Sanders', 'MA')
get_searches('Warren', 'MA')