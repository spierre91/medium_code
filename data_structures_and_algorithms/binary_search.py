def my_binary_search_function(input_list, low_value, high_value,  target_value):
    if type(input_list)!=type([]):
        print('Not a list!')
    else:
        if high_value >= low_value:
            middle = (high_value + low_value) // 2
            if input_list[middle] < target_value:
                return my_binary_search_function(input_list, middle +1, high_value, target_value)
            elif input_list[middle] > target_value:
                return my_binary_search_function(input_list, low_value, middle-1, target_value)
            else:
                return middle
        else:
            return -1
        
        
my_binary_search_function('this_is_a_string', 0, 0, 2)

my_list = [100, 3000, 4500, 5000, 6000, 6050, 7020, 8400, 9100]

my_value = 800

my_result = my_binary_search_function(my_list, 0, len(my_list)-1, my_value) 
  
if my_result != -1: 
    print("{} is present at index".format(my_value), str(my_result)) 
else: 
    print("{} is not present in array".format(my_value)) 
