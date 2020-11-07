import pandas as pd 
import numpy as np 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


df = pd.read_csv('winemag-data-130k-v2.csv')
del df['Unnamed: 0']

print(df.head())



def get_category_mean(categorical_column, categorical_value, numerical_column):
    df_mean = df[df[categorical_column] == categorical_value]
    mean_value = df_mean[numerical_column].mean()
    print(f"Mean {numerical_column} for {categorical_value}: ", np.round(mean_value, 2))
    
get_category_mean('country', 'Italy', 'price')

get_category_mean('variety', 'Pinot Noir', 'price')

def groupby_category_mean(categorical_column, numerical_column):
    df_groupby = df.groupby(categorical_column)[numerical_column].mean()
    df_groupby = np.round(df_groupby,2)
    print(df_groupby)
    
groupby_category_mean('country', 'price')

groupby_category_mean('variety', 'price')


def groupby_category_mode(categorical_column1, categorical_column2):
    df_groupby = df.groupby(categorical_column1)[categorical_column2].agg(pd.Series.mode)
    print(df_groupby)

groupby_category_mode('country', 'variety')
