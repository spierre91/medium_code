class FacebookUser:
    def __init__(self, first_name, last_name, friends_list): 
        self.first_name = first_name
        self.last_name = last_name
        self.friends_list = friends_list
    
    def print_name(self):
           print(f'User is {self.first_name} {self.last_name}')    
    def print_friends(self):
           print(f"User's Friends List: {self.friends_list}")
User1 = FacebookUser('Mike', 'Tyson', ['Buster Douglas', 'Evander Holyfied', 'Roy Jones Jr.'])
           
User1.print_name()


User1.print_friends()



class SpotifyUser(FacebookUser):
    def __init__(self, first_name, last_name, friends_list, playlist):
        super().__init__(first_name, last_name, friends_list)
        self.playlist = playlist
        
    def get_playlist(self):
            print(f"{self.first_name} {self.last_name}'s playlist: {self.playlist}")
            
User2 = SpotifyUser("Floyd", "Mayweather", ['Buster Douglas', 'Evander Holyfied', 'Roy Jones Jr.'], ["Harry Styles", "Taylor Swift"])        

User2.print_name()

User2.print_friends()

User2.get_playlist()
