import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# Step 1 - Load data

books = pd.read_csv('data/goodreads_books.csv', on_bad_lines='skip')

# Step 2 - Data Preprocessing

# Remove spaces in cols
new_cols = [col.strip() for col in books.columns] 
books.columns = new_cols

# Combine authors and title into 1 "content" field
books['content'] = books['authors'].str.lower() + ' ' + books['title'].str.lower() 

# Step 3 - Build Recommender

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['content']) # tfidf of 'content' colun
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Series maps titlees to indices
books['title'] = books['title'].str.strip().str.lower()
indices = pd.Series(books.index, index=books['title'].str.lower()).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    title = title.strip().lower() # normalize
    if title not in indices:
        print(f'{title} not found in dataset\n')
        close = get_close_matches(title, indices.index, n=3, cutoff=0.6) # find matches that it might be
        if close:
            print('Did you mean: ', ", ".join(close), '\n')
            return
        return
    
    idx = indices[title]
    idx = idx if not isinstance(idx, pd.Series) else idx.iloc[0] # pick the first match
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True) # sort books based on similarity in desc order
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]

    return books['title'].iloc[book_indices].tolist()

recs = get_recommendations('great expectations')

with open('outputs/title.txt', 'w') as f:
    for i, rec in enumerate(recs, 1):
        f.write(f"{i}. {rec}\n")
