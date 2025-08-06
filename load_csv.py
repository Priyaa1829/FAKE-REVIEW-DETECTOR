import pandas as pd

# Load CSV file
df = pd.read_csv("reviews.csv")

# Verify the data
print(df.head())  # Display first few rows