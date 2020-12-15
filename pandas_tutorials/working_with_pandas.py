import pandas as pd
import seaborn as sns 

df = pd.read_csv("Bank_churn_modelling.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


#data selection
print(df.head())

df_select = df[['CreditScore', 'Gender', 'Age',  'Exited']]
print(df_select.head())

df_age_gt_40 = df[df['Age'] > 40]
print(df_age_gt_40.head())

df_age_lte_40 = df[df['Age'] <=40]
print(df_age_lte_40.head())


df_age_e_40 = df[df['Age'] ==40]
print(df_age_e_40.head())




df_france = df[df['Geography'] == 'France']
print(df_france.head())



df_france_loc = df.loc[df.Geography == 'France']
print(df_france_loc.head())


geography_list = ['Germany', 'Spain']
df_germany_spain = df[df['Geography'].isin(geography_list)]
print(df_germany_spain.head())


#statistics
mean_credit_score = df['CreditScore'].mean()
print('Mean credit Score: ', mean_credit_score)
std_credit_score = df['CreditScore'].std()
print('Standard Deviation in Credit Score: ', std_credit_score)


min_credit_score = df['CreditScore'].min()
print('Min credit Score: ', min_credit_score)
max_credit_score = df['CreditScore'].max()
print('Max Credit Score: ', max_credit_score)


corr = df[['Age', 'CreditScore', 'EstimatedSalary', 'Tenure']].corr()
print(corr)

sns.heatmap(corr)


#data aggregation
df_groupby_mean = df.groupby('Geography')['CreditScore'].mean()
print(df_groupby_mean.head())

df_groupby_std = df.groupby('Geography')['CreditScore'].std()
print(df_groupby_std.head())


df_groupby_age_mean = df.groupby('Geography')['Age'].mean()
print(df_groupby_age_mean.head())
df_groupby_age_std = df.groupby('Geography')['Age'].std()
print(df_groupby_age_std.head())


df_groupby_multiple_category = df.groupby(['Geography', 'Gender'])['Age'].mean()
print(df_groupby_multiple_category.head())
