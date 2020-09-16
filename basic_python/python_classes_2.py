#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:38:25 2020

@author: sadrachpierre
"""

class Instagram_User:
    def __init__(self, user_name, name, email, private):
        self.user_name = user_name
        self.name = name
        self.email = email
        self.private = private
    def isPrivate(self):
        if self.private:
            print("{} has a Private Account".format(self.name))
        else:
            print("{} has a Public Account".format(self.name))
        
insta_user_1 = Instagram_User('nychef100', 'Jake Cohen', 'jcohen100@gmail.com', True)
insta_user_2 = Instagram_User('worldtraveler123', 'Maria Lopez', 'mlopez123@gmail.com', False)        
        

print(insta_user_1.email)
print(insta_user_2.email)

insta_user_1.isPrivate()
insta_user_2.isPrivate()

'''

insta_user_1 = Instagram_User()
insta_user_2 = Instagram_User()

print("User Object 1: ", insta_user_1)
print("User Object 2: ", insta_user_2)

insta_user_1.user_name = 'nychef100'
insta_user_2.user_name = 'worldtraveler123'

insta_user_1.name = 'Jake Cohen'
insta_user_2.name = 'Maria Lopez'

insta_user_1.email = 'jcohen100@gmail.com'
insta_user_2.email = 'mlopez123@gmail.com'

insta_user_1.private = True
insta_user_2.private = False

print("User Name of user 1: ", insta_user_1.user_name)
print("User Name of user 2: ", insta_user_2.user_name)

'''