
def product(x, y):
    return x*y


print("Product:",product(5, 10))

product_value = lambda x, y: x*y

print("Product w/ Lambda Expression:",product_value(5,10))


import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv("MoviesOnStreamingPlatforms_updated.csv")

del df['Unnamed: 0']
print(df.head())


df = df.assign(rt_ratings = lambda x: x['Rotten Tomatoes'].str.rstrip('%') ) 
  
print(df.head())

df = df.fillna(0)
df = df.assign(round_imdb = lambda x: x['IMDb'].astype(int))
print(df.head())
