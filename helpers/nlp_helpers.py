# helpers/nlp_helpers.py

import spacy

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print("⚠️ spaCy model loading failed:", e)
    nlp = None

def extract_symptoms(text):
    """
    Extract symptom-related entities from user input using spaCy.
    """
    if not nlp:
        return []

    doc = nlp(text)
    symptoms = []

    for ent in doc.ents:
        if ent.label_ in ["SYMPTOM", "DISEASE", "ORG", "NORP", "GPE"]:
            symptoms.append(ent.text.lower())

    return list(set(symptoms))


def extract_medicine_names(text):
    """
    Extract product/medicine-related entities from text.
    """
    if not nlp:
        return []

    doc = nlp(text)
    medicines = []

    for ent in doc.ents:
        if ent.label_ in ["PRODUCT", "ORG", "GPE"]:
            medicines.append(ent.text.lower())

    return list(set(medicines))
