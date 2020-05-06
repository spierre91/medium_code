import pandas as pd
from xml.etree.ElementTree import parse
document = parse('books.xml')

print(document)
print(dir(document))

for item in document.iterfind('book'):
    print(item)

for item in document.iterfind('book'):
    print(item.findtext('author'))

for item in document.iterfind('book'):
    print(item.findtext('title'))
    
    
for item in document.iterfind('book'):
    print(item.findtext('price'))
    
author = []
title = []
price = []    
genre = []
description = []
publish_date = []
for item in document.iterfind('book'):
    author.append(item.findtext('author'))
    title.append(item.findtext('title'))
    price.append(item.findtext('price'))
    genre.append(item.findtext('genre'))
    description.append(item.findtext('description'))
    publish_date.append(item.findtext('publish_date'))
    
df = pd.DataFrame({'title': title, 'author':author, 'price':price,
                   'price':price, 'genre':genre, 'publish_date':publish_date})
print(df)
df['price'] = df['price'].astype(float)
print("Mean price: ", df['price'].mean())


df['publish_date'] = pd.to_datetime(df['publish_date'])
df['year'] = df['publish_date'].dt.year
df['month'] = df['publish_date'].dt.month
df['day'] = df['publish_date'].dt.day

print(df.head())


from collections import Counter
print(Counter(df['author']))
print(Counter(df['genre']))
