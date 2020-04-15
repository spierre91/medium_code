import re

text1 = "python is amazing. I love python, it is the best language. python is the most readable language."
text1 = text1.replace('python', 'C++')
print(text1)
text2 = "The stable release of python 3.8 was on 02/24/2020. The stable release of C++17 was on 12/01/2017."


print(re.sub(r'(\d+)/(\d+)/(\d+)', r'\3-\1-\2', text2))


from calendar import month_abbr
date_pattern = re.compile(r'(\d+)/(\d+)/(\d+)')


def format_date(date_input):
    month_name = month_abbr[int(date_input.group(1))]
    return '{} {} {}'.format(date_input.group(2), month_name, date_input.group(3))
print(date_pattern.sub(format_date, text2))


text3 = "Python is amazing. I love python, it is the best language. Python is the most readable language."

print(text3.replace('python', 'C++'))
print(re.sub('python', 'C++', text3, flags =re.IGNORECASE))
