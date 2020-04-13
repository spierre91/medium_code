my_string = 'python java sql c++ ruby'

print(my_string.split())

my_string2 = 'python java, sql,c++;             ruby'

print(my_string2.split())
import re
print(re.split(r'[;,\s]\s*', my_string))

my_string3 = 'python| java, sql|c++;             ruby'

print(re.split(r'[;|,\s]\s*', my_string3))

my_url = 'http://kaggle.com'

print(my_url.startswith('http:'))
print(my_url.endswith('org'))
print(my_url.endswith('com'))

my_directory = ['python_program.py', 'cpp_program.cpp', 'linear_regression.py', 'text.txt', 'data.csv']
my_scripts = [script for script in my_directory if script.endswith(('.py', '.cpp'))]
print(my_scripts)
