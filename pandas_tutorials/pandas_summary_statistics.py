import pandas as pd 
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("MoviesOnStreamingPlatforms_updated.csv")
del df['Unnamed: 0']
print(df.head())

print("Mean IMDb Rating: ", df['IMDb'].mean())

print("Mean IMDb Rating: ", np.round(df['IMDb'].mean(), 1))

print("Standard Deviation in IMDb Rating: ", np.round(df['IMDb'].std()))
print("Median IMDb Rating: ", np.round(df['IMDb'].median(), 1))
print("Max IMDb Rating: ", df['IMDb'].max())
print("Min IMDb Rating: ", df['IMDb'].min())

def get_statistics(column_name):
    df_copy = df.copy()
    if column_name == 'Rotten Tomatoes':
        df_copy[column_name] = df[column_name].str.rstrip('%')
        df_copy[column_name] = df_copy[column_name].astype(float)
    print(f"Mean {column_name}: ", np.round(df_copy[column_name].mean(), 1))
    print(f"Standard Deviation in {column_name}: ", np.round(df_copy[column_name].std()))
    print(f"Median {column_name}: ", np.round(df_copy[column_name].median(), 1))
    print(f"Max {column_name}: ", df_copy[column_name].max())
    print(f"Min {column_name}: ", df_copy[column_name].min())
    
get_statistics('IMDb')
get_statistics('Runtime')
print(df['Rotten Tomatoes'].head())
get_statistics('Rotten Tomatoes')

print(df[['IMDb', 'Runtime']].describe())

df['Rotten Tomatoes'] = df['Rotten Tomatoes'].str.rstrip('%')
df['Rotten Tomatoes'] = df['Rotten Tomatoes'].astype(float)
print(df[['IMDb', 'Runtime', 'Rotten Tomatoes']].describe())

runtime_genre = df[["Genres", "Runtime"]].groupby("Genres").mean()
print(runtime_genre.head())

rottentomatoes_country = df[["Country", "Rotten Tomatoes"]].groupby("Country").mean().dropna()
print(rottentomatoes_country.head())

def get_group_statistics(categorical, numerical):
    group_df = df[[categorical, numerical]].groupby(categorical).mean().dropna()
    print(group_df.head())
    
get_group_statistics('Genres', 'Runtime')
