import requests

r  = requests.get('https://imgur.com/')
print(r)

print(dir(r))

print(help(r))

print(r.text)

r = requests.get('https://i.imgur.com/1A7VXBR.jpg')
print(r.content)

with open('tree.png', 'wb') as f:
    f.write(r.content)
