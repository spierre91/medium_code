import pandas as pd 

df = pd.read_csv("all_sources_metadata_2020-03-13.csv")

pd.set_option('display.max_columns', None)
print(df.head())

from collections import Counter
print(Counter(df['journal']))


print("List of columns:")
print(list(df.columns))
print("Length of columns:", len(df.columns))

print("Number of rows:", len(df))

df['journal'].dropna(inplace = True)
df.reset_index(inplace = True)

print(Counter(df['journal']).most_common(10))
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

bar_plot = dict(Counter(df['journal'].values).most_common(5))
plt.bar(*zip(*bar_plot.items()))
plt.show()

print(len(set(df['journal'])))

print()
df = df[df.journal == 'Emerg Infect Dis']
print(len(df))

print(list(df['abstract'].values)[0])
print('-'*170)
print(list(df['abstract'].values)[1])
print('-'*170)
print(list(df['abstract'].values)[2])
print('-'*170)
print(list(df['abstract'].values)[3])
print('-'*170)
print(list(df['abstract'].values)[4])
print('-'*170)

from textblob import TextBlob

abstract_sentence = "Clinical trials indicate that taxol is effective in the treatment of patients with refractory ovarian cancer, breast cancer, malignant melanoma and probably other solid tumors."
sentiment_score = TextBlob(abstract_sentence).sentiment.polarity
print("Sentiment Polarity Score:", sentiment_score)


abstract_sentence = "Clinical trials indicate that taxol is ineffective in the treatment of patients with refractory ovarian cancer, breast cancer, malignant melanoma and probably other solid tumors."
sentiment_score = TextBlob(abstract_sentence).sentiment.polarity
print("Sentiment Polarity Score:", sentiment_score)

df['abstract'] = df['abstract'].astype(str)
df['sentiment'] = df['abstract'].apply(lambda abstract: TextBlob(abstract).sentiment.polarity)

df = df[['abstract', 'sentiment']]
print(df.head())

df_pos = df[df['sentiment'] > 0.5]
df_neg = df[df['sentiment'] < 0.0]
print("Number of Positive Results", len(df_pos))
print("Number of Negative Result", len(df_neg))

print(list(df_pos['abstract'].values)[0])
print('-'*170)
print(list(df_pos['abstract'].values)[1])
print('-'*170)
print(list(df_pos['abstract'].values)[2])
print('-'*170)
print(list(df_pos['abstract'].values)[3])
print('-'*170)
print(list(df_pos['abstract'].values)[4])
print('-'*170)


df_neg = df[df['sentiment'] < -0.1]
print(list(df_neg['abstract'].values)[0])
print('-'*170)
print(list(df_neg['abstract'].values)[1])
print('-'*170)
print(list(df_neg['abstract'].values)[2])
print('-'*170)
print(list(df_neg['abstract'].values)[3])
print('-'*170)
print(list(df_neg['abstract'].values)[4])
print('-'*170)