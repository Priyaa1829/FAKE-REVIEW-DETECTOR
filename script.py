from datasets import load_dataset
import pandas as pd

# Load dataset from Hugging Face
ds = load_dataset("theArijitDas/Fake-Reviews-Dataset")

# Convert to DataFrame (Use 'train' split)
df = pd.DataFrame(ds["train"])

# Save DataFrame as a CSV file
df.to_csv("reviews.csv", index=False)  

print("✅ Dataset saved as reviews.csv")