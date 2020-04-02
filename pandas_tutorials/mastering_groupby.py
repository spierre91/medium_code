import pandas as pd 

df = pd.read_csv("2019-nCoV-cases-JHU.csv")



df = df[df['Region'] == 'US']

df.reset_index(inplace = True)
del df['index']

df = df[df['Province'] != 'Unassigned Location (From Diamond Princess)']
df['State'] = df['Province'].str[-2:]

df['County'] = df['Province'].str[:-4]

df['Date'] = pd.to_datetime(df['Date'], format ='%m/%d/%Y %H:%M')




gb = df.groupby('State')
print(type(gb))

df_ca = gb.get_group('CA').set_index('Date')
print(set(df_ca.index))
print(df_ca.head())


deaths = pd.DataFrame()

for name, group in df.groupby('State'):
    if deaths.empty:
        deaths = group.set_index('Date')[["Deaths"]].rename(columns={"Deaths": name})
    else:
        deaths = deaths.join(group.set_index('Date')[["Deaths"]].rename(columns={"Deaths": name}))
del deaths['go']
del deaths['ia']
del deaths['is'] 
del deaths['na']
del deaths['on'] 
del deaths['s)']     
deaths.fillna(0, inplace = True)
print(deaths.tail())     

death_stats = deaths.describe()
death_stats = death_stats.astype(int)
print(death_stats)


def get_statistics(column_name):
    column_df = pd.DataFrame()
    
    for name, group in df.groupby('State'):
        if column_df.empty:
            column_df = group.set_index('Date')[[column_name]].rename(columns={column_name: name})
        else:
            column_df = column_df.join(group.set_index('Date')[[column_name]].rename(columns={column_name: name}))
    del column_df['go']
    del column_df['ia']
    del column_df['is'] 
    del column_df['na']
    del column_df['on'] 
    del column_df['s)']     
    column_df.fillna(0, inplace = True)    
    
    column_stats = column_df.describe()
    column_stats = column_stats.astype(int)
    print(column_stats)

get_statistics("Confirmed")
get_statistics("Recovered")


df['Day'] = df['Date'].dt.day
def get_statistics_day(column_name):
    column_df = pd.DataFrame()
    
    for name, group in df.groupby('State'):
        if column_df.empty:
            column_df = group.set_index('Day')[[column_name]].rename(columns={column_name: name})
        else:
            column_df = column_df.join(group.set_index('Day')[[column_name]].rename(columns={column_name: name}))

    column_df.fillna(0, inplace = True)    
    
    del column_df['go']
    del column_df['ia']
    del column_df['is'] 
    del column_df['na']
    del column_df['on'] 
    del column_df['s)']   
    column_stats = column_df.describe()
    column_stats = column_stats.astype(int)
    print(column_stats)
    
get_statistics_day("Confirmed")
