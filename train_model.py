import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import nltk
import string

# Download stopwords (only once)
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
data = pd.read_csv('reviews.csv')

# Clean text function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply cleaning
data['Review'] = data['Review'].apply(clean_text)
data['Label'] = data['Label'].map({'FAKE': 0, 'REAL': 1})  # Convert labels

# Features and labels
X = data['Review']
y = data['Label']

# Vectorize
vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Model and vectorizer saved successfully!")
