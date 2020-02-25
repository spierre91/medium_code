#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:18:29 2020

@author: sadrachpierre
"""

import pandas as pd 


s1 = pd.Series(['python is awesome. I love python.', 'java is just ok. I like python more', 
                'C++ is overrated. You should consider learning another language, like Java or Python.'], dtype="string")

print(s1)

s2 = pd.Series(['100', 'unknown', '20', '240', 'unknown', '100'], dtype="string")
print(s2.str.match('100'))

print(s2.str.match('un'))
print(s2.str.isdigit())

s_upper = s1.str.upper()

print(s_upper)

s_lower = s_upper.str.lower()

print(s_lower)


print(s1.str.len())


s3 = pd.Series([' python', 'java', 'ruby', 'fortran'])

print(s3)
print(s3.str.strip())

print(s3.str.lstrip())


print(s3.str.strip())



s4 = pd.Series([' python\n', 'java\n', 'ruby \n', 'fortran \n'], dtype='string')
print(s4)


print(s4.str.strip(' \n'))




s5 = pd.Series(['$#1200', 'dollar1,000', 'dollar10000', '$500'], dtype="string")
print(s5)                
s5 = s5.str.replace('#', '')               
s5 = s5.str.replace('dollar', '$')
s5 = s5.str.replace(',', '')
print(s5)
                   
                