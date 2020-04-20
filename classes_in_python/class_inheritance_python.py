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
            
user_1.isPremium()
user_2.isPremium()
            
class Premium_User(Spotify_User):
    def __init__(self, subscription_tier, name, email, premium):
        self.subscription_tier = subscription_tier
        super(Premium_User, self).__init__(name, email, premium)
    def premium_content(self):
        if self.subscription_tier == 'Tier 1':
            print("Music streaming with no advertisements")
        if self.subscription_tier == 'Tier 2':
            print("Tier 1 content + Live Concert Streaming")
user3 = Premium_User('Tier 1', 'Megan Harris', 'mharris@gmail.com', True)    
user3.isPremium()
user3.premium_content()  

user4 = Premium_User('Tier 2', 'Bill Rogers', 'brogers@gmail.com', True)
user4.premium_content()
