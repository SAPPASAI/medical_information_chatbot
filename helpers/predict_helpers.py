import pandas as pd
import numpy as np
import joblib 
import re
import ast
from difflib import get_close_matches #fuzzy matching

# Load the trained model and label encoder
model = joblib.load("model/model.pkl")
le = joblib.load("model/label_encoder.pkl")
X_columns = joblib.load("model/X_columns.pkl")  # list of symptom feature names

# Load all data files
desc_df = pd.read_csv("data/description.csv")
medications_df = pd.read_csv("data/medications.csv")
precautions_df = pd.read_csv("data/precautions_df.csv")
workout_df = pd.read_csv("data/workout_df.csv")
diets_df = pd.read_csv("data/diets.csv")

# Clean disease columns
desc_df['Disease'] = desc_df['Disease'].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
medications_df['Disease'] = medications_df['Disease'].astype(str).str.strip().str.lower()
precautions_df['Disease'] = precautions_df['Disease'].astype(str).str.strip().str.lower()
workout_df.columns = workout_df.columns.str.strip().str.lower()
diets_df.columns = diets_df.columns.str.strip().str.lower()

# Create a valid symptom list for validation
valid_symptoms = {
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety',
 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin',
 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain',
 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm',
 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness',
 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum',
 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples',
 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze', 'prognosis'
}

def predict_disease(symptom_list):
    input_vector = [0] * len(X_columns)
    symptoms = [col.lower() for col in X_columns]

    valid_input_found = False
    for symptom in symptom_list:
        # Try direct match first
        if symptom in valid_symptoms:
            match = symptom
        else:
            # Try fuzzy matching
            matches = get_close_matches(symptom, valid_symptoms, n=1, cutoff=0.7)
            if matches:
                match = matches[0]
                print(f"🔍 Interpreting '{symptom}' as '{match}'")
            else:
                print(f"⚠️ Warning: '{symptom}' is not recognized.")
                continue

        if match in symptoms:
            input_vector[symptoms.index(match)] = 1
            valid_input_found = True

    if not valid_input_found:
        return None  # No valid symptom found

    input_df = pd.DataFrame([input_vector], columns=X_columns)
    prediction = model.predict(input_df)[0]
    return le.inverse_transform([prediction])[0]


def get_description(disease_name):
    try:
        disease_name_clean = disease_name.strip().lower().replace("👉", "").strip()
        disease_name_clean = re.sub(r'\s+', ' ', disease_name_clean)
        desc_row = desc_df[desc_df['Disease'] == disease_name_clean]
        if not desc_row.empty:
            return desc_row.iloc[0]['Description']
        else:
            return "No description available for this disease."
    except Exception as e:
        return f"❌ Error reading description: {e}"

def get_medications(disease_name):
    try:
        result = medications_df[medications_df['Disease'] == disease_name.lower()]
        if not result.empty:
            meds_string = result.iloc[0]['Medication']
            medications = ast.literal_eval(meds_string)
            return [med.strip() for med in medications if med.strip()]
        else:
            return ["No medication information available for this disease."]
    except KeyError as e:
        return [f"⚠️ Column not found: {e}"]
    except Exception as e:
        return [f"❌ Error fetching medication: {e}"]

def get_precautions(disease):
    try:
        row = precautions_df[precautions_df['Disease'] == disease.lower()]
        if not row.empty:
            precautions = row.iloc[0, 1:].dropna().tolist()
            return precautions
        else:
            return ["No precautions found for this disease."]
    except Exception as e:
        return [f"❌ Error fetching precautions: {e}"]

def get_workouts(disease):
    try:
        rows = workout_df[workout_df['disease'].str.lower() == disease.lower()]
        if not rows.empty:
            workouts = rows['workout'].dropna().tolist()
            return workouts
        else:
            return ["No workout recommendations available for this disease."]
    except Exception as e:
        return [f"Error fetching workouts: {e}"]

def get_diets(disease):
    try:
        row = diets_df[diets_df['disease'].str.lower() == disease.lower()]
        if not row.empty:
            diet_str = row.iloc[0, 1]  # Assuming the second column contains the diet string
            diets = ast.literal_eval(diet_str)  # Safely convert string to list
            return [d.strip() for d in diets if d.strip()]
        else:
            return ["No diet recommendations available for this disease."]
    except Exception as e:
        return [f"Error fetching diets: {e}"]
