import pandas as pd

books = pd.read_csv('data/goodreads_books.csv', on_bad_lines='skip')
books.columns = [col.strip() for col in books.columns]

# Drop ID-like and non-numeric fields
drop_cols = ["bookID", "isbn", "isbn13", "title", "authors", "publisher", "language_code", "publication_date"]
books_numeric = books.drop(columns=drop_cols, errors="ignore")
books_numeric = books_numeric.apply(pd.to_numeric, errors="coerce")

corr_matrix = books_numeric.corr()

corr_pairs = (
    corr_matrix.unstack()
    .reset_index()
    .rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: "Correlation"})
)

# Remove self correlations
corr_pairs = corr_pairs[corr_pairs['Feature 1'] != corr_pairs['Feature 2']]
corr_pairs["Pair"] = corr_pairs.apply(lambda x: tuple(sorted((x["Feature 1"], x["Feature 2"]))), axis=1)
corr_pairs = corr_pairs.drop_duplicates(subset="Pair").drop(columns="Pair")

# Sort by absolute correlation value
corr_pairs = corr_pairs.reindex(corr_pairs.Correlation.abs().sort_values(ascending=False).index)

top_corr = corr_pairs.drop_duplicates(subset=['Correlation'])

top_corr = corr_pairs.head(10)

corr_pairs.to_csv("outputs/top_correlations.csv", index=False)