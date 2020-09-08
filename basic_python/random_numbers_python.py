#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:03:57 2020

@author: sadrachpierre
"""

import random 
bmi_list = [29, 18, 20, 22, 19, 25, 30, 28,22, 21, 18, 19, 20, 20, 22, 23]
print("First random choice:", random.choice(bmi_list))
print("Second random choice:", random.choice(bmi_list))
print("Third random choice:", random.choice(bmi_list))

print("Random sample, N=5 :", random.sample(bmi_list, 5))

print("Random sample, N=10:", random.sample(bmi_list, 10))

print("BMI list: ", bmi_list)
random.shuffle(bmi_list)
print("Shuffled BMI list: ", bmi_list)

print("Random Integer: ", random.randint(1,5))


random_ints_list = []
for i in range(1,50):
    n = random.randint(1,5)
    random_ints_list.append(n)
print("My random integer list: ", random_ints_list)


print("Random Float: ", random.random())

random_float_list = []
for i in range(1,10):
    n = random.random()*500
    if n>=100.0:
        random_float_list.append(n)
print("My random float list: ", random_float_list)


import numpy as np
uniform_list = np.random.uniform(-10,1,50)

print("Uniformly Distributed Numbers: ", uniform_list)

normal_list = np.random.uniform(-50,0,50)
print("Normally Distributed Numbers: ", normal_list)




