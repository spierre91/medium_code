from google.oauth2 import service_account
from google.cloud import bigquery

key_path = "key.json"
credentials = service_account.Credentials.from_service_account_file(
    key_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
client = bigquery.Client(
    credentials=credentials,
    project=credentials.project_id,
)

dataset_ref = client.dataset("medicare", project="bigquery-public-data")


columns = [x.table_id for x in client.list_tables(dataset_ref)]
print(columns)

table_ref = dataset_ref.table('inpatient_charges_2011')
table = client.get_table(table_ref)

df = client.list_rows(table).to_dataframe()
print(df.columns)

print("Length: ", len(df))
from collections import Counter
print(dict(Counter(df['provider_name'].values).most_common(20)))

import matplotlib.pyplot as plt
import seaborn as sns 

def plot_most_common(category):
    bar_plot = dict(Counter(df[category].values).most_common(4))
    sns.set()
    plt.bar(*zip(*bar_plot.items()))
    plt.show()

#plot_most_common('drg_definition')

def get_boxplot_of_categories(data_frame, categorical_column, numerical_column, limit):

    keys = []
    for i in dict(Counter(df[categorical_column].values).most_common(limit)):
        keys.append(i)
    print(keys)
    
    df_new = df[df[categorical_column].isin(keys)]
    sns.boxplot(x = df_new[categorical_column], y = df_new[numerical_column])
    
#get_boxplot_of_categories(df, 'provider_name', 'total_discharges', 5)
    
def get_scatter_plot_category(data_frame, categorical_column, categorical_value, numerical_column_one, numerical_column_two):
    import matplotlib.pyplot as plt
    import seaborn as sns
    df_new = data_frame[data_frame[categorical_column] == categorical_value]
    sns.set()
    plt.scatter(x= df_new[numerical_column_one], y = df_new[numerical_column_two])
    plt.xlabel(numerical_column_one)
    plt.ylabel(numerical_column_two)
    
get_scatter_plot_category(df, 'provider_name', 'NORTH SHORE MEDICAL CENTER', 'average_medicare_payments', 'total_discharges')
