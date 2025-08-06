import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load Feature-Engineered Data
X, y = pickle.load(open("vectorized_data.pkl", "rb"))

# Step 2: Split Data into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Step 4: Evaluate Model Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model training complete! Accuracy: {accuracy:.2f}")

# Step 5: Save the Trained Model
with open("fake_review_detector.pkl", "wb") as file:
    pickle.dump(model, file)
print("✅ Model saved as fake_review_detector.pkl!")