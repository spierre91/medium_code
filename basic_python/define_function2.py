
def sum_function(first, second):
    return (first + second)

print("Sum: ", sum_function(5, 10))


def sum_arbitrary_arguments(first, *rest):
    return (first + sum(rest))

print("Sum with arbitrary number of arguments: ", sum_arbitrary_arguments(5, 10,15))
print("Sum with arbitrary number of arguments: ", sum_arbitrary_arguments(5, 10,15, 20, 25, 30))


def print_weather(name, state = "New York", temp = "30 degrees Celsius"):
   print("Hi {}, it is {} in {}".format(name, temp, state))


print_weather("Sarah")
print_weather("Mike", "Boston", "28 degrees Celsius")


def print_weather_kwargs(**kwargs):
    weather_report = ""
    for argument in kwargs.values():
        weather_report += argument
    return weather_report
kwargs =  { 'temp':"30 degrees Celsius ", 'in':'in', 'State':" New York", 'when': ' today'}
print(print_weather_kwargs(**kwargs))
