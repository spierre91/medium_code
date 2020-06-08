news_dict = {"News":["Selling Toilet Paper During the Pandemic", 
                     "How to Reopen Gyms?", "Covid-19 Risk Based on Blood Type?"] ,
             "Clicks":[100, 500, 10000] }

print(news_dict)
print(dir(news_dict))

#news_dict.clear()
print("Cleared dictionary: ", news_dict)

copy_dict = news_dict.copy()


copy_dict['Clicks'] = [100, 100, 100]
print(news_dict)
print(copy_dict)
print(news_dict['News'])
print(news_dict['Clicks'])



print(news_dict.get("News"))
print(news_dict.get("Clicks"))
print(news_dict.items())


for key, value in news_dict.items():
    print(key, value)
    
print(news_dict.keys())



news_dict.pop('Clicks')
print(news_dict)

news_dict.update(News = "New York City Begins to Reopen", Clicks = 30000)
print(news_dict.values())
