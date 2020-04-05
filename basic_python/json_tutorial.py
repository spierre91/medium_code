'''
{
     "album_title":"Yellow Submarine",     
     "release_year":1966 ,     
     "won_grammy": false,
     "band": "The Beatles",
     "musicians": ["John Lennon", "Paul McCartney", "George Harrison", "Ringo Starr"],
     "studio": {"studio_name": "Abbey Road Studios", "location": "London, England"}
     }
'''

import json

print(dir(json))

album_json_file = open("album.txt", "r")

album = json.load(album_json_file)

album_json_file.close()

print(album)

print("Object type: ", type(album))

print("Album Title: ", album['album_title'])

print("Release Year: ", album['release_year'])



album_s_json_file = """{"album_title" : "Yellow Submarine",
     "release_year" : 1966,
     "won_grammy" : false,
     "band" : "The Beatles",
     "album_sale": null,
     "musicians" : ["John Lennon", "Paul McCartney", "George   Harrison", "Ringo Starr"],
     "studio" : {"studio_name": "Abbey Road Studios", "location": "London, England"}
     }"""
album_s = json.loads(album_s_json_file)
print(album_s)

album = {'album_title': 'Yellow Submarine', 'release_year': 1966, 'won_grammy': False, 
 'band': 'The Beatles', 'album_sale': None, 'musicians': ['John Lennon', 'Paul McCartney', 'George   Harrison', 'Ringo Starr'], 
 'studio': {'studio_name': 'Abbey Road Studios', 'location': 'London, England'}}

print(json.dumps(album))
print(type(json.dumps(album)))
