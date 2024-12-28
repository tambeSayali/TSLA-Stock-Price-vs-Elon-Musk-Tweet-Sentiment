import pandas as pd

# Load the CSV file
df = pd.read_csv('combined_tweets.csv')

# Check for duplicates in the entire dataset
duplicates = df[df.duplicated()]

# Or check for duplicates in specific columns, e.g., 'column_name'
# duplicates = df[df.duplicated(['column_name'])]

# Print the duplicates if any
print(duplicates)

# Count duplicates
print(f"Number of duplicates: {duplicates.shape[0]}")
