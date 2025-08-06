import pandas as pd
import os

# Check if the file exists
file_path = "clean_reviews.csv"
if os.path.exists(file_path):
    print("✅ clean_reviews.csv found!")
else:
    print("⚠️ Error: clean_reviews.csv is missing!")

# Load and display the cleaned dataset
df = pd.read_csv(file_path)
print(df.head())  # Show first few rows
print(df.columns) # Show column names
print(df.shape)   # Show number of rows and columns
