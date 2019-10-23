import pandas as pd  
import tweepy
pd.set_option('display.max_rows', 10000000)
pd.set_option('display.max_columns', 1000000)

consumer_key = '' 
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)



def get_related_tweets(key_word):
    twitter_users = []
    tweet_time = []
    tweet_string = [] 
    for tweet in tweepy.Cursor(api.search,q=key_word, count=1000).items(1000):
            if (not tweet.retweeted) and ('RT @' not in tweet.text):
                if tweet.lang == "en":
                    twitter_users.append(tweet.user.name)
                    tweet_time.append(tweet.created_at)
                    tweet_string.append(tweet.text)
                    #print([tweet.user.name,tweet.created_at,tweet.text])
    df = pd.DataFrame({'name':twitter_users, 'time': tweet_time, 'tweet': tweet_string})
    
    return df 


df_bad = get_related_tweets("Joker bad movie")
print(df_bad.head())
for i in range(0,10):
    print(df_bad['tweet'].iloc[i])
    
    
df_good = get_related_tweets("Joker good movie")
print(df_good.head(5))
for i in range(10,20):
    print(df_good['tweet'].iloc[i])
