import re
sentence1 = 'Python is great'



result1 = re.search("^Python.*great$", sentence1)
print(result1)


result2 = re.search("^Python.*bad$", sentence1)
print(result2)


text_list = ['Python is great', 'C plus plus is nice', 'Python is a fantastic language, it is great', 
             'Java is a nice programming language' ]

for text in text_list:
    print(re.search("^Python.*great$", text))
    
    
for text in text_list:
    result = re.search("^Python.*great$", text)
    if result:
        print('"', text, '"', 'Begins with Python and ends with great')
    else:
        print('"', text, '"', 'Does not begin with Python and end with great')
        


text_list = ['Python is great', 'C plus plus is nice', 'Python is a fantastic language, it is great', 
             'Java is a nice programming language' ]        
        

def match_begin_end(begin, end):        
    for text in text_list:
        text_lower = text.lower()
        begin_lower = begin.lower()
        end_lower = end.lower()
        result = re.search(f"^{begin_lower}.*{end_lower}$", text_lower)
        if result:
            print('"', text, '"', f'Begins with {begin_lower} and ends with {end_lower}')
        else:
            print('"', text, '"', f'Does not begin with {begin_lower} and end with {end_lower}')

match_begin_end('Python', 'great')




#case sensitive
match_begin_end('C plus plus', 'nice')
match_begin_end('c plus plus', 'nice')



def match_begin(begin):        
    for text in text_list:
        text_lower = text.lower()
        begin_lower = begin.lower()
        result = re.search(f"^{begin_lower}", text_lower)
        if result:
            print('"', text, '"', f'Begins with {begin_lower}')
        else:
            print('"', text, '"', f'Does not begin with {begin_lower}')
match_begin('Python')
            
            
            

def match_end(end):        
    for text in text_list:
        text_lower = text.lower()
        end_lower = end.lower()
        result = re.search(f"{end_lower}$", text_lower)
        if result:
            print('"', text, '"', f'Ends with {end_lower}')
        else:
            print('"', text, '"', f'Does not end with {end_lower}')
match_end('great')


def match_either(str_one, str_two):        
    for text in text_list:
        text_lower = text.lower()
        str_one_lower = str_one.lower()
        str_two_lower = str_two.lower()
        result = re.search(f"{str_one_lower}|{str_two_lower}", text_lower)
        if result:
            print(f"'{str_one_lower}' or '{str_two_lower}' found in", text)
        else:
            print(f"Neither '{str_one_lower}' nor '{str_two_lower}' found in", text)
match_either('c plus plus', 'nice')
