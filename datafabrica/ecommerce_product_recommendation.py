import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("ecommerce_data_electronics.csv")

print(df.head())

df['time_stamp'] = pd.to_datetime(df['time_stamp'])

NOW = df['time_stamp'].max()
rfmTable = df.groupby('card_name').agg({'time_stamp': lambda x: (NOW - x.max()).days, 'product_id': lambda x: len(x), 'price': lambda x: x.sum()})
rfmTable['time_stamp'] = rfmTable['time_stamp'].astype(int)
rfmTable.rename(columns={'time_stamp': 'recency', 
                         'product_id': 'frequency',
                         'price': 'monetary_value'}, inplace=True)


print(rfmTable.head())



rfmTable['r_quartile'] = pd.qcut(rfmTable['recency'], q=4, labels=range(1,5), duplicates='raise')
rfmTable['f_quartile'] = pd.qcut(rfmTable['frequency'], q=4, labels=range(1,5), duplicates='drop')
rfmTable['m_quartile'] = pd.qcut(rfmTable['monetary_value'], q=4, labels=range(1,5), duplicates='drop')
rfm_data = rfmTable.reset_index()

print(rfm_data.head())

rfm_data['r_quartile'] = rfm_data['r_quartile'].astype(str)
rfm_data['f_quartile'] = rfm_data['f_quartile'].astype(str)
rfm_data['m_quartile'] = rfm_data['m_quartile'].astype(str)


rfm_data['RFM_score'] = rfm_data['r_quartile'] + rfm_data['f_quartile'] + rfm_data['m_quartile']
print(rfm_data.head())


rfm_data['customer_segment'] = 'Other'

rfm_data.loc[rfm_data['RFM_score'].isin(['334', '443', '444', '344', '434', '433', '343', '333']), 'customer_segment'] = 'Premium Customer' #nothing <= 2
rfm_data.loc[rfm_data['RFM_score'].isin(['244', '234', '232', '332', '143', '233', '243']), 'customer_segment'] = 'Repeat Customer' # f >= 3 & r or m >=3
rfm_data.loc[rfm_data['RFM_score'].isin(['424', '414', '144', '314', '324', '124', '224', '423', '413', '133', '323', '313', '134']), 'customer_segment'] = 'Top Spender' # m >= 3 & f or m >=3
rfm_data.loc[rfm_data['RFM_score'].isin([ '422', '223', '212', '122', '222', '132', '322', '312', '412', '123', '214']), 'customer_segment'] = 'At Risk Customer' # two or more  <=2
rfm_data.loc[rfm_data['RFM_score'].isin(['411','111', '113', '114', '112', '211', '311']), 'customer_segment'] = 'Inactive Customer' # two or more  =1




from collections import Counter 

print(Counter(rfm_data['customer_segment']))


def generate_recommendations(target_customer, cohort, num_recommendations=5):
    user_item_matrix = cohort.groupby('card_name')['product'].apply(lambda x: ', '.join(x)).reset_index()
    user_item_matrix['product_descriptions'] = cohort.groupby('card_name')['product_description'].apply(lambda x: ', '.join(x)).reset_index()['product_description']
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(user_item_matrix['product_descriptions'])
    similarity_matrix = cosine_similarity(tfidf_matrix)    
    target_customer_index = user_item_matrix[user_item_matrix['card_name'] == target_customer].index[0]
    similar_customers = similarity_matrix[target_customer_index].argsort()[::-1][1:num_recommendations+1]
    target_customer_purchases = set(user_item_matrix[user_item_matrix['card_name'] == target_customer]['product'].iloc[0].split(', '))

    recommendations = []
    for customer_index in similar_customers:
        customer_purchases = set(user_item_matrix.iloc[customer_index]['product'].split(', '))
        new_items = customer_purchases.difference(target_customer_purchases)
        recommendations.extend(new_items)
    
    return list(set(recommendations))[:num_recommendations]


rfm_data = rfm_data[rfm_data['customer_segment']== 'Premium Customer']
premium = list(set(rfm_data['card_name']))[:10]
df_premium = df[df['card_name'].isin(premium)]

print(df_premium.head())
recommendations = generate_recommendations("Ashley Perry", df_premium, num_recommendations=5)

print("Recommendations for Ashley Perry:")
print(recommendations)


recommendations = generate_recommendations("Clifford Stanley", df_premium, num_recommendations=5)

print("Recommendations for Clifford Stanley")
print(recommendations)
