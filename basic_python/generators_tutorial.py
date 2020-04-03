def generate_list(input_value):
    number, numbers = 0, []
    while number <= input_value:
        numbers.append(number)
        number += 2
    return numbers

values  = generate_list(10)
print(generate_list(10))
print("Sum of list: ", sum(values))
class iterator_object(object):
    def __init__(self, input_value):
        self.input_value = input_value
        self.number = 0
        
    def __iter__(self):
        return self
    def __next__(self):
       return self.next()
   
    def next(self):
        if self.number <= self.input_value:
            current, self.number = self.number, self.number + 2
            return current
        else:
            raise StopIteration()
            
value = iterator_object(10)
print("Sum using an Iterator Object: ", sum(value))

def generator_function(input_value):
    number = 0
    while number <= input_value:
        yield number
        number += 2
value = generator_function(10)
print("Sum using a Generator: ", sum(value))
