#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:01:03 2020

@author: sadrachpierre
"""

new_run = ['09/01/2020', '10:00', 60, 6, 100]
date, pace, time, distance, elevation = new_run

print("Date: ", date)
print("Pace: ", pace, 'min')
print("Time: ", time, 'min')
print("Distance: ", distance, 'miles')
print("Elevation: ", elevation, 'feet')

new_run_with_splits = ['09/01/2020', '10:00', 60, 6, 100, 
                       ('12:00', '12:00', '10:00', '11:00', '8:00', '7:00')]


date, pace, time, distance, elevation, splits = new_run_with_splits

print("Mile splits: ", splits)


date, pace, time, distance, elevation, (mile_1, mile_2, mile_3, mile_4, mile_5, mile_6) = new_run_with_splits
print("Mile One: ", mile_1)

print("Mile Two: ", mile_2)
print("Mile THree: ", mile_3)
print("Mile Four: ", mile_4)
print("Mile Five: ", mile_5)
print("Mile Six: ", mile_6)


date, pace, time, distance, elevation, (first, *middle, last) = new_run_with_splits


print("First: ", first)
print("Middlle: ", middle)
print("Last: ", last)


new_run_with_splits_dict = ['09/01/2020', '10:00', 60, 6, 100, {'mile_1': '12:00', 'mile_2':'12:00', 'mile_3':'10:00', 'mile_4':'11;00', 'mile_5':'8:00', 'mile_6':'7:00'}]

date, pace, time, distance, elevation, splits_dict = new_run_with_splits_dict


print("Mile One: ", splits_dict['mile_1'])
print("Mile Two: ", splits_dict['mile_2'])
print("Mile Three: ", splits_dict['mile_3'])
