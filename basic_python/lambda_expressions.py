def subtract(x, y):
    return x - y

print("Difference:", subtract(10, 7))

subtract_value = lambda x, y: x - y
print("Difference:",subtract_value(10,7))


names = ['Guido van Rossum', 'Bjarne Stroustrup', 'James Gosling', 'Rasmus Lerdorf']
print(sorted(names, key = lambda name : name.split()[-1].lower()))

x = 5
add = lambda y: x + y
print("Addition:", add(24))

x= 10
print("Addition:", add(24))

x = 5
add = lambda y,x=x: x + y
print("Addition:", add(24))
x= 10
print("Addition:", add(24))
