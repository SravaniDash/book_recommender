import pandas as pd

# Step 1 - Explore the data
books = pd.read_csv('goodreads_books.csv', on_bad_lines='skip')
print(books.head())
print(books.info())
print(books.isnull().sum())
