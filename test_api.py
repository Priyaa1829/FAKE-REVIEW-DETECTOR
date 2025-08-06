import pickle

# Load trained model and vectorizer
model = pickle.load(open("fake_review_detector.pkl", "rb"))
vectorizer = pickle.load(open("check_vectorizer.pkl", "rb"))

# Sample review for testing
test_review = ["buy our products"]

# Convert text to numerical features
review_vector = vectorizer.transform(test_review)

# Get prediction
prediction = model.predict(review_vector)[0]

print("Prediction:", "Fake" if prediction == 0 else "Real")