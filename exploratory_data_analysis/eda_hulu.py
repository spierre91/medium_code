import pandas as pd 
df = pd.read_csv("MoviesOnStreamingPlatforms_updated.csv")

print(list(df.columns))
del df['Unnamed: 0']
print(df.head())


df_hulu = df[df['Hulu'] == 1]
print("Total Length: ", len(df))
print("Hulu Length: ", len(df_hulu))


print(df_hulu.head())


def return_counter(data_frame, column_name, limit):
   from collections import Counter    
   print(dict(Counter(data_frame[column_name].values).most_common(limit)))


return_counter(df_hulu, 'Language', 5)
return_counter(df_hulu, 'Genres', 5)

df_d1 = df_hulu[df_hulu['Genres'] =='Documentary']
print(set(df_d1['Title']))

print(set(df_d1['Country']))

print(set(df_d1['Runtime']))



def return_statistics(data_frame, categorical_column, numerical_column):
    mean = []
    std = []
    field = []
    for i in set(list(data_frame[categorical_column].values)):
        new_data = data_frame[data_frame[categorical_column] == i]
        field.append(i)
        mean.append(new_data[numerical_column].mean())
        std.append(new_data[numerical_column].std())
    df = pd.DataFrame({'{}'.format(categorical_column): field, 'mean {}'.format(numerical_column): mean, 'std in {}'.format(numerical_column): std})
    df.sort_values('mean {}'.format(numerical_column), inplace = True, ascending = False)
    df.dropna(inplace = True)
    return df




stats = return_statistics(df_hulu, 'Genres', 'Runtime')
print(stats.head(15))

def get_boxplot_of_categories(data_frame, categorical_column, numerical_column, limit):
    import seaborn as sns
    from collections import Counter
    keys = []
    for i in dict(Counter(df[categorical_column].values).most_common(limit)):
        keys.append(i)
    print(keys)
    df_new = df[df[categorical_column].isin(keys)]
    sns.set()
    sns.boxplot(x = df_new[categorical_column], y =      df_new[numerical_column])
    

#get_boxplot_of_categories(df, 'Genres', 'Runtime', 5)



def get_histogram(data_frame, numerical_column):
    df_new = data_frame
    df_new[numerical_column].hist(bins=100)
    
    
get_histogram(df_hulu, 'Runtime')  
