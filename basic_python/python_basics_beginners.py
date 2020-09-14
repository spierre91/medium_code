#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:26:46 2020

@author: sadrachpierre
"""

names = ['Bob', 'Sarah', 'Ted', 'Nicole']
print("Names: ", names)

heights = [180.0, 160.0, 190.0, 150.0]
print("Heights: ", heights)

names_height = {'Bob':180.0, 'Sarah':160.0, 'Ted':190.0, 'Nicole':150.0}
print("Names & Heights: ", names_height)

names_height_2 = dict(zip(names, heights))
print("Names & Heights using dict() & zip(): ", names_height_2)

heights_feet = [] #initialize empty list
for height in heights:
    heights_feet.append(height/30.48)
print("Height in Feet: ", heights_feet)


heights_feet_2 = [height/30.48 for height in heights]
print("Height in Feet List Comprehension: ", heights_feet_2)


names_height_feet = {name:height/30.48 for name, height in zip(names,heights)}
print("Names & Heights in Feet Dictionary Comprehension: ", names_height_feet)



def convert_to_feet(height_cm_dict):
   height_feet_dict = {}
   for key, value in height_cm_dict.items():
       height_feet_dict[key] = value/30.48
   print("Names & Heights in Feet Function Conversion: ", height_feet_dict)
   
   
convert_to_feet(names_height)