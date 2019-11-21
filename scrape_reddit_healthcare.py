#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:50:09 2019

@author: sadrachpierre
"""

import praw
import pandas as pd
#NOTE: these credential aren't valid
r = praw.Reddit(client_id = 'AbCd1234!', 
                     client_secret = '4321dCbA!', 
                     username= 'username101',
                     password= 'password101!',
                     user_agent='someagentinfo ')







def get_post(topic):
    title = []
    ups = []
    downs = []
    visited = []
    selftext = []
    time_list = []
    subreddit = r.subreddit(topic)
    for submission in subreddit.hot(limit = None):
        if not submission.stickied:
            title.append(submission.title)
            ups.append(submission.ups)
            downs.append(submission.downs)
            visited.append(submission.visited)
            selftext.append(submission.selftext)
            time_list.append(datetime.datetime.fromtimestamp(submission.created))
    
        
    df = pd.DataFrame({'Title': title, 'selftext': selftext, 'ups': ups, 'downs': downs, 
                       'visited': visited, 'time': time_list})
    return df


df_aetna = get_post('aetna')
print(df_aetna.head())


df_medicare = get_post('medicare')
print(df_medicare.head())

df_aetna.to_csv("aetna.csv")
df_medicare.to_csv("medicare.csv")