best_albums = ["Sgt. Pepper's Lonely Hearts Club Band", "Pet Sounds", "Revolver", "Highway 61 Revisited",  "Rubber Soul"]

for album in best_albums:
    print(album)
    
print(dir(best_albums))

print(next(best_albums))

iter_best_albums = best_albums.__iter__()

print(dir(iter_best_albums))


print(next(iter_best_albums))
print(next(iter_best_albums))
print(next(iter_best_albums))



while True:
    try:
        element = next(iter_best_albums)
        print(element)
    except(StopIteration):
        break
            
