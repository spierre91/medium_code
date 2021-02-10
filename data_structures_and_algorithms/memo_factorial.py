factorial_dict = {}
def factorial_memo(input_value):
    if input_value < 2: 
        return 1
    if input_value not in factorial_dict:
        factorial_dict[input_value] = input_value * factorial_memo(input_value-1)
    return factorial_dict[input_value]
    


for i in range(1, 5000):
     print(f"{i}! = ", factorial_memo(i))
     
     
from functools import lru_cache
@lru_cache(maxsize = 1000)
def factorial(input_value):
    if input_value < 2: 
        return 1
    elif input_value >= 2:
        return input_value * factorial(input_value-1)
    
    
for i in range(1, 5000):
     print(f"{i}! = ", factorial(i))
