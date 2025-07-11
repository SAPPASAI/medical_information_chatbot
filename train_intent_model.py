import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Expanded training data with varied medicine info queries
training_data = [
    # Greetings
    ("hi", "greeting"), ("hello", "greeting"), ("hey", "greeting"),
    ("good morning", "greeting"), ("good evening", "greeting"),
    ("yo bot", "greeting"), ("hi assistant", "greeting"),
    ("greetings", "greeting"), ("how are you", "greeting"),

    # Farewells
    ("bye", "farewell"), ("goodbye", "farewell"), ("see you", "farewell"),
    ("take care", "farewell"), ("catch you later", "farewell"),
    ("exit", "farewell"), ("logging out", "farewell"),

    # Thanks
    ("thank you", "thanks"), ("thanks a lot", "thanks"),
    ("cheers", "thanks"), ("much appreciated", "thanks"),
    ("you're great", "thanks"), ("thank you so much", "thanks"),

    # Composition Queries
    ("composition of dolo 650", "medicine_query"),

    # Uses / Benefits Queries
    ("uses of dolo 650", "medicine_query"),
    ("what is dolo 650 used for", "medicine_query"),
    ("benefits of azithromycin", "medicine_query"),

    # Side Effects Queries
    ("side effects of dolo 650", "medicine_query"),
    ("what are the side effects of sinarest", "medicine_query"),

    # Class Queries
    ("action class of ibuprofen", "medicine_query"),
    ("is paracetamol herbal or chemical", "medicine_query"),

    # Habit Forming Queries
    ("is alprazolam habit forming", "medicine_query"),
    ("is this medicine addictive", "medicine_query"),
    ("habit forming nature of lorazepam", "medicine_query"),

    # How to Use Queries
    ("how to use dolo 650", "medicine_query"),

    # Product Info Queries
    ("tell me about sinarest", "medicine_query"),
    ("information about calpol", "medicine_query"),

    # Symptom Check
    ("I have a fever", "symptom_check"),
    ("I feel headache and sore throat", "symptom_check"),
    ("I have chills and body ache", "symptom_check"),
    ("nausea and dizziness", "symptom_check"),
    ("my stomach hurts", "symptom_check"),
    ("I have chest pain", "symptom_check"),
    ("shortness of breath", "symptom_check"),
    ("I can't sleep and I feel anxious", "symptom_check"),
    ("I have a skin rash", "symptom_check"),
    ("I feel dizzy", "symptom_check"),
    ("I have a migraine", "symptom_check"),
    ("I feel tired all the time", "symptom_check"),
    ("I have pain in my abdomen", "symptom_check"),
    ("I have sore muscles", "symptom_check"),
    ("my throat hurts", "symptom_check"),
    ("I can’t stop coughing", "symptom_check"),
    ("I feel bloated", "symptom_check"),
    ("I feel pressure in my head", "symptom_check"),
    ("my legs feel numb", "symptom_check"),
    ("my eyes are itchy", "symptom_check"),
    ("I feel cold even in warm weather", "symptom_check"),
    ("I can’t breathe properly", "symptom_check"),
    ("my back hurts", "symptom_check")
]

# Prepare training data
X = [text for text, _ in training_data]
y = [label for _, label in training_data]

# Vectorization and model training
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)
model = LogisticRegression()
model.fit(X_vec, y)

# Save model and vectorizer
model_path = "intent_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump((model, vectorizer), f)

model_path
