#manually set attributes
class Spotify_User:
    pass

user_1 = Spotify_User()
user_2 = Spotify_User()

user_1.name = 'Sarah Phillips'
user_2.name = 'Todd Grant'

user_1.email = 'sphillips@gmail.com'
user_2.email = 'tgrant@gmail.com'

user_1.premium = True
user_2.premium = False


print(user_1)
print(user_2)

print(user_1.name)
print(user_2.name)

#set attributes use initialization method
#define an additional method
class Spotify_User:
    def __init__(self, name, email, premium):
        self.name = name
        self.email = email
        self.premium = premium
        
    def isPremium(self):
        if self.premium:
            print("{} is a Premium User".format(self.name))
        else:
            print("{} is not a Premium User".format(self.name))
user_1 = Spotify_User('Sarah Phillips', 'sphillips@gmail.com', True)
user_2 = Spotify_User('Todd Grant', 'tgrant@gmail.com', False)        

print(user_1.email)
print(user_2.email)

user_1.isPremium()
user_2.isPremium()
