import re

test_string1 = 'Python is Amazing!'

regex_1 = re.findall(r"^\w+",test_string1)
print(regex_1)

test_string2 = 'Java is Amazing!'

regex_2 = re.findall(r"^\w+",test_string2)
print(regex_2)

youtube_titles = [("How to Tell if We're Beating COVID-19", 2200000), ("Extreme Closet Clean Out",326000), ("This is $1,000,000 in Food",8800000),
                    ("How To Tell If Someone Truly Loves You ", 2800000), ("How to Tell Real Gold from Fake", 2300000), ("Extreme living room transformation ", 25000),
                    ]
first_words = []
views = []
for title in youtube_titles:
    first_words.append(re.findall(r"^\w+",title[0])[0])
    views.append(title[1])
    
print(first_words)
print(views)

for title in youtube_titles:
    print(re.findall(r"^\w+",title[0])[0])
    
import pandas as pd
df = pd.DataFrame({'first_words': first_words, 'views':views})
df = df.groupby('first_words')['views'].mean().sort_values(ascending = False)
print(df)
    
