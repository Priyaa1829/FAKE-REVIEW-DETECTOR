from flask import Flask, render_template_string, request
import joblib

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Flask app
app = Flask(__name__)

# HTML UI using Flask render_template_string
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Fake Review Detector</title>
</head>
<body style="text-align: center; padding: 50px; font-family: sans-serif;">
    <h1>🕵️ Fake Review Detector</h1>
    <form method="POST">
        <textarea name="review" rows="5" cols="60" placeholder="Enter a review..."></textarea><br><br>
        <input type="submit" value="Check Review">
    </form>
    {% if prediction %}
        <h2>Prediction: <span style="color:{{ color }}">{{ prediction }}</span></h2>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    color = "black"
    if request.method == 'POST':
        review = request.form['review']
        vec = vectorizer.transform([review])
        result = model.predict(vec)[0]
        prediction = "FAKE REVIEW" if result == 1 else "GENUINE REVIEW"
        color = "red" if result == 1 else "green"
    return render_template_string(HTML, prediction=prediction, color=color)

if __name__ ==  '__main__':
    app.run(debug=True)
