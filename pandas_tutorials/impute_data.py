import pandas as pd 
df = pd.read_csv("winemag-data-130k-v2.csv")
del df['Unnamed: 0']
print(df.head())

print(df.info())
print(type(df.isnull().sum()))

from collections import Counter
print(Counter(df['country']))


df_US = df[df['country']=='US']
print(df_US.isnull().sum())

def impute_numerical(categorical_column, numerical_column):
    frames = []
    for i in list(set(df[categorical_column])):
        df_category = df[df[categorical_column]== i]
        if len(df_category) > 1:    
            df_category[numerical_column].fillna(df_category[numerical_column].mean(),inplace = True)        
        else:
            df_category[numerical_column].fillna(df[numerical_column].mean(),inplace = True)
        frames.append(df_category)    
        final_df = pd.concat(frames)
    return final_df 


def impute_categorical(categorical_column1, categorical_column2):
    cat_frames = []
    for i in list(set(df[categorical_column1])):
        df_category = df[df[categorical_column1]== i]
        if len(df_category) > 1:    
            df_category[categorical_column2].fillna(df_category[categorical_column2].mode()[0],inplace = True)        
        else:
            df_category[categorical_column2].fillna(df[categorical_column2].mode()[0],inplace = True)
        cat_frames.append(df_category)    
        cat_df = pd.concat(cat_frames)
    return cat_df 


print(df.isnull().sum())    
impute_price  = impute_numerical('country', 'price')
print(impute_price.isnull().sum())


impute_taster = impute_categorical('country', 'taster_name')
print(impute_taster.isnull().sum())
