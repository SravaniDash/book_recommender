import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1 - Load data

books = pd.read_csv('goodreads_books.csv', on_bad_lines='skip')

# Step 2 - Data Preprocessing

# Remove spaces in cols
new_cols = [col.strip() for col in books.columns] 
books.columns = new_cols

# Combine authors and title into 1 "content" field
books['content'] = books['authors'].str.lower() + ' ' + books['title'].str.lower() 

# Step 3 - Build Recommender

