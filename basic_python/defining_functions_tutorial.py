import pandas as pd 
def print_quote():
    return "If they can get you asking the wrong questions, they don't have to worry about the answers."\

quote = print_quote()

print(quote)

def quote_length(input_quote):
    return len(input_quote)

quote_one = "The most difficult challenge to the ideal is its transformation into reality, and few ideals survive."
quote_two = "We've had the goddamn Age of Faith, we've had the goddamn Age of Reason. This is the Age of Publicity."
quote_three = "Reading Proust isn't just reading a book, it's an experience and you can't reject an experience."


print(quote_length(quote_one))
print(quote_length(quote_two))
print(quote_length(quote_three))


quote_list = [quote_one, quote_two, quote_three]
def combine_quote(string_list, author_name):
   print("Author: ", author_name)
   return ''.join(string_list)


print(combine_quote(quote_list, 'William Gaddis'))


quote_list = ["If they can get you asking the wrong questions, they don't have to worry about the answers.", "The most difficult challenge to the ideal is its transformation into reality, and few ideals survive.", "The truth will set you free. But not until it is finished with you."]
book_list = ["Gravity's Rainbow", "The Recognitions", "Infinite Jest"] 
author_list = ["Thomas Pynchon", "William Gaddis", "David Foster Wallace"]
number_of_pages = [776, 976, 1088]



def get_dataframe(quote, book, author, pages):
    df = pd.DataFrame({'quotes':quote,   'books':book, 'authors':author, 'pages':pages})
    return df, len(df), df['books'], df['quotes']


df, length, books, quotes = get_dataframe(quote_list, book_list, author_list, number_of_pages)
print(books)

print(quotes)
