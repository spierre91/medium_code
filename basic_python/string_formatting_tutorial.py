def get_message_pct(name, email, age, height, weight):
    print("%s's email is %s. He is %s years old, %s feet tall, and weighs %s."%(name, email,  age, height, weight))
    
#get_message_pct('John', 'johnadams@gmail.com', 50,  "6",  '180 pounds')

def get_message_format(name, email, age, height, weight):
    print("{}'s email is {}. He is {} years old, {} feet tall, and weighs {}.".format(name, email,  age, height, weight))
    

get_message_format('John', 'johnadams@gmail.com', 50,  "6",  '180 pounds')


jake_info = {'name': 'Jake', 'email': 'jake@gmail.com', 'age':30, 'height':"6", 'weight':'170 pounds'}
get_message_format(**jake_info)



def get_message_fstrings(name, email, age, height, weight):
    print(f"{name}'s email is {email}. She is {age} years old, {height} feet tall, and weighs {weight}.")
    
get_message_fstrings('Sarah', 'sarah@yahoo.com', 20, '5', '120 pounds')
