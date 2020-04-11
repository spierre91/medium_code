class YouTube:
    def __init__(self, channel_name, subscribers, verification):
        self.channel_name = channel_name
        self.subscribers = subscribers
        self.verification = verification
        
    def __repr__(self):
        return 'YouTube({0.channel_name!r}, {0.subscribers!r}, {0.verification!r})'.format(self)
    
    def __str__(self):
        return '({0.channel_name!s}, {0.subscribers!s}, {0.verification!s})'.format(self)

sentdex = YouTube('Sentdex', 847000, True)

print(sentdex)



sentdex = YouTube('Sentdex', 847000, True)
print('sentdex is {0!r}'.format(sentdex))

print('sentdex is {0}'.format(sentdex))
