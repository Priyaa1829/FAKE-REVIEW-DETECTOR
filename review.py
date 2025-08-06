import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Step 1: Load raw dataset
df = pd.read_csv("reviews.csv")

# Step 2: Clean text (remove special characters, numbers, extra spaces)
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters
    text = text.lower().strip()  # Convert to lowercase
    return text

df["text"] = df["text"].apply(clean_text)

# Step 3: Remove Stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
df["text"] = df["text"].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))

# Step 4: Ensure labels are numeric (0 for fake, 1 for real)
df["label"] = df["label"].map({"fake": 0, "real": 1})  # Convert text labels to numbers

# Step 5: Remove any missing or empty rows
df.dropna(inplace=True)

# Step 6: Save cleaned data
df.to_csv("clean_reviews.csv", index=False)

print("✅ Data preprocessing complete! Saved as clean_reviews.csv")