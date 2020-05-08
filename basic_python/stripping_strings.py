
string1 = '     Python is an experiment in how much freedom programmers need. \n'
print(string1)
print(string1.strip())


string2 = "     Too much freedom and nobody can read another's code; too little and expressiveness is endangered. \n\n\n"
print(string2)
print(string2.lstrip())
print(string2.rstrip())

string3 = "#####Too much freedom and nobody can read another's code; too little and expressiveness is endangered.&&&&"
print(string3)
print(string3.lstrip('#'))
print(string3.rstrip('&'))
print(string3.strip('#&'))
                    
                    
string4 = "&&&&&&&Too much freedom and nobody can read another's code; &&&&&&& too little and expressiveness is endangered.&&&&&&&"
print(string4)
print(string4.strip('&'))
print(string4.replace('&', ''))
