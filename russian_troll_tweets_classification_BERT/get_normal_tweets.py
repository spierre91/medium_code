#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 12:36:47 2019

@author: sadrachpierre
"""

import tweepy
consumer_key = 'abc123'
consumer_secret = 'abc123'
access_token = 'abc123'
access_token_secret = 'abc123'


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)



import pandas as pd  
pd.set_option('display.max_rows', 10000000)
pd.set_option('display.max_columns', 1000000)



def get_related_tweets(key_word):

    twitter_users = []
    tweet_time = []
    tweet_string = [] 
    for tweet in tweepy.Cursor(api.search,q=key_word, count=1000).items(1000):
                if tweet.lang == "en":
                    twitter_users.append(tweet.user.name)
                    tweet_time.append(tweet.created_at)
                    tweet_string.append(tweet.text)
                    #print([tweet.user.name,tweet.created_at,tweet.text])
    df = pd.DataFrame({'name':twitter_users, 'time': tweet_time, 'tweet': tweet_string})
    
    return df 


df = get_related_tweets('ff')
df.to_csv('ff.csv')


df = get_related_tweets('followfriday')
df.to_csv('followfriday.csv')


df = get_related_tweets('tuesdaymotivation')
df.to_csv('tuesdaymotivation.csv')

df = get_related_tweets('thankful')
df.to_csv('thankful.csv')

df = get_related_tweets('birthday')
df.to_csv('birthday.csv')

df = get_related_tweets('pet')
df.to_csv('pet.csv')

df = get_related_tweets('funny')
df.to_csv('funny.csv')

df = get_related_tweets('influencer')
df.to_csv('influencer.csv')
    
df1 = pd.read_csv('ff.csv')
df2 = pd.read_csv('followfriday.csv')
df3 = pd.read_csv('tuesdaymotivation.csv')
df4 = pd.read_csv('thankful.csv')
df5 = pd.read_csv('birthday.csv')
df6 = pd.read_csv('pet.csv')
df7 = pd.read_csv('funny.csv')
df8 = pd.read_csv('influencer.csv')

df_full = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], axis=0)
df_full.to_csv("normal_tweets.csv")
print(len(df_full))


import pandas as pd 

df_bots = pd.read_csv('tweets.csv')
df_normal = pd.read_csv('normal_tweets.csv')

df_bots['type'] = 'bot'
df_normal['type'] = 'normal'
df_normal['text'] = df_normal['tweet']
df_bots = df_bots[['text', 'type']]
df_normal = df_normal[['text', 'type']]


df  = df_normal.append(df_bots)
df = df.sample(frac=1, random_state = 24).reset_index(drop=True)
df.to_csv("training_data.csv")
