class AppleMusicUser:
    def __init__(self, apple_id, plays):
        self.apple_id = apple_id
        self.plays = plays 
    def __repr__(self):
        return 'AppleID({}, plays:{})'.format(self.apple_id, self.plays)
        
user1 = AppleMusicUser("emusk@tesla.com", None)
print(user1)

users = [AppleMusicUser("mzuckerberg@facebook.com", 100), AppleMusicUser("jdorsey@twitter.com", 20), 
         AppleMusicUser("emusk@tesla.com", 50)]

print(users)
print(sorted(users, key = lambda u: u.apple_id))

from operator import attrgetter
print(sorted(users, key = attrgetter('plays')))
