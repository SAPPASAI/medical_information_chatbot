import sqlite3
import pickle
from helpers import predict_helpers as ph
from helpers import medicine_helpers as mh
from helpers import example_medicine_helper as emh
from helpers import nlp_helpers as nlp
import pandas as pd
from difflib import get_close_matches
import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load ML intent model
try:
    model_path = os.path.join(BASE_DIR, "intent_model.pkl")
    intent_model, intent_vectorizer = pickle.load(open(model_path, "rb"))
except Exception as e:
    print("⚠️ Error loading intent model:", e)

# Load medicine data
try:
    excel_path = os.path.join(BASE_DIR, 'assets', 'MID.xlsx')
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip().str.lower()
    df["name"] = df["name"].astype(str).str.strip().str.lower()
except FileNotFoundError:
    print("❌ Error: MID.xlsx not found. Make sure it's in the 'backend/assets/' directory.")


def classify_intent(text):
    try:
        vec = intent_vectorizer.transform([text])
        return intent_model.predict(vec)[0]
    except:
        return "general"


def extract_medicine_name(query):
    keywords = ["uses", "side effects", "composition", "manufacturer", "how to use",
                "how does it work", "benefits", "safety", "habit forming", "class"]
    cleaned_query = query.lower()
    
    for keyword in keywords:
        cleaned_query = cleaned_query.replace(keyword, "").strip()

    cleaned_query = re.sub(r"\b(what|is|the|of|tell|me|about|effects|are|side|"
                           r"manufacture|please|information|details|give|tablet|"
                           r"capsule|syrup|medicine|med)\b", "", cleaned_query)
    
    cleaned_query = re.sub(r"[^a-zA-Z0-9\s]", "", cleaned_query).strip()
    return cleaned_query


def find_best_match(name):
    names = df["name"].dropna().tolist()
    extracted = extract_medicine_name(name)

    if extracted in names:
        return extracted

    matches = get_close_matches(extracted, names, n=1, cutoff=0.6)
    return matches[0] if matches else None


def get_info_type(query):
    query = query.lower()
    if any(kw in query for kw in ["how to use", "how do i take", "usage", "use"]):
        return "howtouse"
    elif any(kw in query for kw in ["side effect", "adverse effect"]):
        return "sideeffect"
    elif "benefit" in query:
        return "productbenefits"
    elif "safety" in query:
        return "safetyadvice"
    elif "habit" in query or "addictive" in query:
        return "habit_forming"
    elif "chemical" in query:
        return "chemical_class"
    elif "therapeutic" in query:
        return "therapeutic_class"
    elif "action" in query:
        return "action_class"
    elif "composition" in query:
        return "contains"
    else:
        return None


def get_bot_response(message, user_id=None):
    message_lower = message.lower().strip()

    # Predict intent
    intent = classify_intent(message_lower)

    # INTENT HANDLING
    if intent == "greeting":
        return "👋 Hello! How can I help you today?"

    elif intent == "thanks":
        return "🙏 You're welcome! Feel free to ask anything!"

    elif intent == "farewell":
        return "👋 Goodbye! Take care of your health!"

    elif intent == "image_request":
        return "📸 Image support is coming soon!"

    elif intent == "medicine_query":
        # Handle alternative medicine queries
        if "alternative" in message_lower:
            med_name = message_lower.replace("alternative", "").strip()
            alternatives = emh.find_alternative_medicines(med_name)
            if alternatives:
                return "💊 Alternative Medicines:\n" + "\n".join(alternatives)
            else:
                return "❌ Sorry, I couldn't find alternatives for that medicine."

        # General or specific medicine queries
        matched_medicine = find_best_match(message_lower)
        if matched_medicine:
            info_type = get_info_type(message_lower)
            result = mh.search_medicine(matched_medicine, info_type)

            if info_type:
                # Return only the requested field
                return result
            else:
                # Return full info + follow-up
                follow_up = (
                    "\n\n👉 Would you like to know more about "
                    f"{matched_medicine.title()}?\nYou can ask about: 'side effects', 'how to use', "
                    "'benefits', 'safety advice', 'chemical class', 'composition'"
                )
                return result + follow_up

        return "❌ Sorry, I couldn't understand. Please enter symptoms (comma separated) or a known medicine name."

    elif intent == "symptom_check":
        symptoms = nlp.extract_symptoms(message)
        if not symptoms:
            symptoms = [s.strip().lower() for s in message.split(",") if s.strip()]

        if len(symptoms) >= 1:
            disease = ph.predict_disease(symptoms)
            if disease:
                description = ph.get_description(disease)
                meds = ph.get_medications(disease)
                precautions = ph.get_precautions(disease)
                workouts = ph.get_workouts(disease)
                diets = ph.get_diets(disease)

                response = f"🩺 **Predicted Disease**: {disease}\n\n"
                response += f"📝 **Description**: {description}\n"
                response += f"💊 **Medications**: {meds}\n"
                response += f"⚠️ **Precautions**: {precautions}\n"
                response += f"🥗 **Diet**: {diets}\n"
                response += f"🏃 **Workouts**: {workouts}"

                # Save prediction to DB
                if user_id:
                    try:
                        conn = sqlite3.connect("medical_chatbot.db")
                        cursor = conn.cursor()
                        cursor.execute(
                            "INSERT INTO Prediction (user_id, symptoms, predicted_disease) VALUES (?, ?, ?)",
                            (user_id, ', '.join(symptoms), disease)
                        )
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        print("[DB ERROR] Failed to insert prediction:", e)

                return response

            return ("🔍 I couldn't identify a disease based on the symptoms provided.\n"
                    "Please provide more details or correct symptoms.")

    elif "my name is" in message_lower or "i am" in message_lower:
        return "I'm a medical chatbot, I don't need to know your name. How can I help with your symptoms or medicine questions?"

    return ("❓ Sorry, I couldn't understand that. Try asking:\n"
            "- 'Tell me about Avastin'\n"
            "- 'Side effects of Andol'\n"
            "- 'How to use Bevacizumab'\n"
            "- Or list your symptoms like 'headache, fever'")
