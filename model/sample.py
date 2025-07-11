import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import joblib
import os

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Load dataset
data = pd.read_csv("model/Training.csv")  # Ensure this file exists

# Separate features and target
X = data.drop(columns=['prognosis'])
y = data['prognosis']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

# Save model
with open("model/model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save label encoder
joblib.dump(le, "model/label_encoder.pkl")

# Save column names for prediction
joblib.dump(X.columns.tolist(), "model/X_columns.pkl")

print("✅ Decision Tree model, LabelEncoder, and X_columns saved in 'model/'")
