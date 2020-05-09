import pandas as pd
pd.set_option('display.max_columns', None)
df = pd.read_csv("../USA_cars_datasets.csv")
del df[list(df.columns)[0]]
print(list(df.columns))

print("Number of rows: ", len(df))
print(df.head())

from collections import Counter    
def return_counter(data_frame, column_name, limit):
   print(dict(Counter(data_frame[column_name].values).most_common(limit)))
   
return_counter(df, 'brand', 5)

return_counter(df, 'color', 5)

df_d1 = df[df['color'] =='white']
print(set(df_d1['brand']))

print(dict(Counter(df_d1['brand']).most_common(5)))

print(dict(Counter(df_d1['state']).most_common(5)))



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



stats = return_statistics(df, 'brand', 'price')
print(stats.head(15))

import matplotlib.pyplot as plt

def get_boxplot_of_categories(data_frame, categorical_column, numerical_column, limit):
    import seaborn as sns
    from collections import Counter
    keys = []
    for i in dict(Counter(df[categorical_column].values).most_common(limit)):
        keys.append(i)
    print(keys)
    df_new = df[df[categorical_column].isin(keys)]
    sns.set()
    sns.boxplot(x = df_new[categorical_column], y = df_new[numerical_column])
    plt.show()
    
get_boxplot_of_categories(df, 'brand', 'price', 5)    
    

def get_histogram(data_frame, numerical_column):
    df_new = data_frame
    df_new[numerical_column].hist(bins=100)
    plt.title('{} histogram'.format(numerical_column))
    plt.show()
get_histogram(df, 'price')
