import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Step 1: Load the raw dataset
df = pd.read_csv("reviews.csv")

# Step 2: Remove special characters, numbers, and extra spaces
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters
    text = text.lower().strip()  # Convert to lowercase
    return text

df["text"] = df["text"].apply(clean_text)

# Step 3: Remove Stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
df["text"] = df["text"].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))

# Step 4: Remove empty rows
df.dropna(inplace=True)

# Step 5: Save cleaned data
df.to_csv("clean_reviews.csv", index=False)
print("✅ Text cleaning complete! Saved as clean_reviews.csv")