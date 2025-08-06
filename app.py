import pickle
from flask import Flask, request, render_template
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model & vectorizer
model = pickle.load(open("fake_review_detector.pkl", "rb"))
vectorizer = pickle.load(open("check_vectorizer.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)
@app.route("/")
def welcome():
    return render_template("welcome.html")


# Route for homepage (renders `index.html`) 
@app.route("/home")
def home():
    return render_template("index.html")

# Route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    review_text = request.form.get("review")  # Safely get form data
    
    if not review_text:
        return jsonify({"error": "No review text provided"}), 400
    
    # Convert text to TF-IDF features
    review_vector = vectorizer.transform([review_text])
    
    # Get prediction from trained model
    prediction = model.predict(review_vector)[0]

    return jsonify({"prediction": "Fake" if prediction == 0 else "Real"})
# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)