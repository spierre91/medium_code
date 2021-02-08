import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('reddit_wsb.csv')

print(list(df.columns))

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df.head())

import numpy as np 

df['GME_title'] = np.where(df['title'].str.contains('GME'), 1, 0)
print(df[['title','GME_title']].head())


df['GME_body'] = np.where(df['body'].str.contains('GME'), 1, 0)
print(df[['title','GME_body']].head())


def return_counter(data_frame, column_name):
    from collections import Counter    
    print(dict(Counter(data_frame[column_name].values)))
print(return_counter(df, 'GME_body'))




df['ticker'] = np.where(df['GME_title'] ==1, 'GME', 'Other')

print(df.head())


df_GME = df[df['GME_title']==1]
print(df_GME.head())






def get_boxplot_of_categories(data_frame, categorical_column, numerical_column):
    import seaborn as sns
    from collections import Counter
    keys = []
    for i in dict(Counter(df[categorical_column].values)):
        keys.append(i)
    print(keys)
    df_new = df[df[categorical_column].isin(keys)]
    sns.set()
    sns.boxplot(x = df_new[categorical_column], y =      df_new[numerical_column])
    plt.show()
    
get_boxplot_of_categories(df, 'ticker', 'score')


def get_histogram(data_frame, numerical_column):
    df_new = data_frame
    df_new[numerical_column].hist(bins=100)
    plt.show()
get_histogram(df, 'score')


from textblob import TextBlob

def get_sentiment(df):
    df['sentiment'] = df['title'].apply(lambda title: TextBlob(title).sentiment.polarity)
    df_pos = df[df['sentiment'] > 0.0]
    df_neg = df[df['sentiment'] < 0.0]
    print("Number of Positive Posts", len(df_pos))
    print("Number of Negative Posts", len(df_neg))
    
    sns.set()
    labels = ['Postive', 'Negative']
    heights = [len(df_pos), len(df_neg)]
    plt.bar(labels, heights, color = 'navy')
    plt.title('GME Posts Sentiment')
    plt.show()
    return df
df = get_sentiment(df_GME)


print(df.head())
