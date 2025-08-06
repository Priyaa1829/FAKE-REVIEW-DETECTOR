import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Step 1: Load Cleaned Data
df = pd.read_csv("clean_reviews.csv")

# Step 2: Ensure text column is properly formatted
df["text"] = df["text"].astype(str)  # Convert all entries to string
df.dropna(subset=["text"], inplace=True)  # Remove any remaining NaN values

# Step 3: Apply TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, min_df=1)  # Use top 5000 words, allow rare words
X = vectorizer.fit_transform(df["text"])  # Convert text to numerical format

# Step 4: Save Vectorizer for Model Training
with open("check_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

# Step 5: Save Feature-Engineered Data
pickle.dump((X, df["label"].values), open("vectorized_data.pkl", "wb"))

print("✅ Step 3 complete! Saved as check_vectorizer.pkl and vectorized_data.pkl")