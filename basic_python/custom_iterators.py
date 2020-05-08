def increment_pattern(start, stop, increment):
    x = start
    while x < stop:
        yield x
        x += increment
        
for value in increment_pattern(0, 3, 0.5):
    print(value)
    
    
def decrement_pattern(start, stop, decrement):
    x = start
    while x > stop:
        yield x
        x -= decrement
        
for value in decrement_pattern(3, 0, 0.5):
    print(value)


def compound_interest_monthly(start, stop, increment):
    x = start
    principle = 1000
    while x < stop:
        interest = (1+ 0.08/12)**(12*x)
        yield principle*interest
        x += increment
        
for value in compound_interest_monthly(0, 19, 1):
    print(value)

def compound_interest_quarterly(start, stop, increment):
    x = start
    principle = 1000
    while x < stop:
        interest = (1+ 0.04/4)**(4*x)
        yield principle*interest
        x += increment
        
for value in compound_interest_quarterly(0, 19, 1):
    print(value)
