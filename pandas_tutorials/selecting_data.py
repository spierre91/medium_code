mport pandas as pd 
pd.set_option('display.max_columns', None)
df = pd.read_csv("vgsales.csv")

print(df.head())
print(list(df.columns))
#
print(df.info())

print(df.head())

df_sports = df.loc[df.Genre == 'Sports']

print(df_sports.head())

df_wii = df.loc[df.Platform == 'Wii']

print(df_wii.head())

df_wii_racing = df_wii.loc[df.Genre == 'Racing']

print(df_wii_racing.head())

df_wii_racing = df.loc[(df.Platform == 'Wii') & (df.Genre == 'Racing')]
print(df_wii_racing.head())

df_gt_1mil = df.loc[(df.Platform == 'Wii') & (df.Genre == 'Racing') & (df.Global_Sales >= 1.0)]
print(df_gt_1mil.head())


df_filter_rows = df.iloc[:1000]
print("Length of original: ", len(df))
print("Length of filtered: ", len(df_filter_rows))

df_random_sample = df.sample(n=5000, random_state = 42)
print("Length of sample: ", len(df_random_sample))

print(df_random_sample.head())
