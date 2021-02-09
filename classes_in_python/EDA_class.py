import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns 

class Summary:
    def __init__(self, data):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        self.df = pd.read_csv(data)
        del self.df['Unnamed: 0']
    def print_head(self, rows):
        print(self.df.head(rows))
    def get_columns(self):
        print(list(self.df.columns))
    def get_dim(self):
        print('Rows:', len(self.df))
        print('Columns:', len(list(self.df.columns)))
    def get_stats(self):
        print(self.df.describe())
    def get_mean(self, column):
        print(f"Mean {column}:", self.df[column].mean())
    def get_standard_dev(self, column):
        print(f"STD {column}:", self.df[column].std())
    def get_counter(self, column):
        print(dict(Counter(self.df[column])))
    def get_hist(self, column):
        sns.set()
        self.df[column].hist(bins=100)
        plt.title(f"{column} Histogram")
        plt.show()
    def get_boxplot_of_categories(self, categorical_column, numerical_column, limit):
        keys = []
        for i in dict(Counter(self.df[categorical_column].values).most_common(limit)):
            keys.append(i)
        df_new = self.df[self.df[categorical_column].isin(keys)]
        sns.set()
        sns.boxplot(x = df_new[categorical_column], y = df_new[numerical_column])
        plt.title(f"Boxplots of {categorical_column} and {numerical_column}")
        plt.show()
        
    def get_heatmap(self, columns):
        df_new = self.df[columns]
        sns.set()
        sns.heatmap(df_new.corr())
        plt.title(f"Heatmap of {columns}")

data = Summary('fifa_data.csv')
data.print_head(10)
data.get_columns()
data.get_dim()
data.get_stats()
data.get_mean('Age')
data.get_standard_dev('Age')
data.get_counter('Nationality')
data.get_hist('Age')
data.get_boxplot_of_categories('Nationality', 'Age', 5)
data.get_heatmap(['Age', 'Overall', 'Potential', 'Special'])
