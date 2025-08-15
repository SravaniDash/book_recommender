import pandas as pd

# Step 1 - Load data
books = pd.read_csv('data/goodreads_books.csv', on_bad_lines='skip')
books.columns = [col.strip() for col in books.columns]

# Step 2 - Load correlations
corrs = pd.read_csv('helper_files/top_correlations.csv')
corrs['abs_corr'] = corrs['Correlation'].abs()
top_corr = corrs.sort_values(by='abs_corr', ascending=False).iloc[0]

feature1 = top_corr['Feature 1']
feature2 = top_corr['Feature 2']

print(f"Strongest correlation is between '{feature1}' and '{feature2}'")

# Step 3 - Recommendation based on strongest correlation
# Ask user for a value of feature1
user_value = float(input(f"Enter a value for {feature1}: "))

# Find closest matches to the user input
books['diff'] = (books[feature1] - user_value).abs()
top_books = books.sort_values(by=['diff', feature2], ascending=[True, False]).head(3)

open('outputs/corrs.txt', 'w').write(
    '\n'.join(f"{i}. {row.title} ({feature1}: {getattr(row, feature1)}, {feature2}: {getattr(row, feature2)})"
              for i, row in enumerate(top_books.itertuples(), 1))
)
