import numpy as np

my_list1 = [1, 2, 3, 4, 5]
for value in my_list1:
    print(value)
    
    
for value in my_list1:
    square = value**2
    print(value, square)
    
    
my_list2 = [1, 2, 3, 4, 5, '6']


for value in my_list2:
    try:
        square = value**2
        print(value, square)
    except(TypeError):
        square = int(value)**2    
        print(value, square)
        
        
my_list3 = [1, 2, 3, 4, 5, '##']

for value in my_list3:
    try:
        square = value**2
        print(value, square)
    except(TypeError):
        square = np.nan
        print(value, square)
            

my_list4 = ['python c++ java',  'SQL R scala', 'pandas keras sklearn', {'language': 'golang ruby julia'}]
for value in my_list4:
    try:
        print([i +' is awesome' for i in value.split()])
    except(AttributeError):
        print([i +' is awesome' for i in value['language'].split()])

