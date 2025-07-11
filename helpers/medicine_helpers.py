import pandas as pd
from difflib import get_close_matches
import re
import os

# Load and normalize medicine data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(BASE_DIR, "..", "assets", "MID.xlsx")

try:
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip().str.lower()
    df["name"] = df["name"].astype(str).str.strip().str.lower()
except Exception as e:
    print("❌ Error loading MID.xlsx:", e)
    df = pd.DataFrame()  # fallback empty

# ----------------------------------------

def extract_medicine_name(query):
    keywords = [
        "uses", "side effects", "composition", "manufacturer", "how to use",
        "how does it work", "benefits", "safety", "habit forming", "class",
        "product information", "info", "information", "details", "tablet", "syrup", "capsule"
    ]
    cleaned = query.lower()
    for kw in keywords:
        cleaned = cleaned.replace(kw, "")

    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", cleaned)
    return cleaned.strip()


def find_best_match(name):
    names = df["name"].dropna().unique().tolist()
    extracted = extract_medicine_name(name)

    if extracted in names:
        return extracted

    matches = get_close_matches(extracted, names, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_info_type(query):
    query = query.lower()

    if any(x in query for x in ["how to use", "how do i take", "usage", "take"]):
        return "howtouse"
    elif any(x in query for x in ["side effect", "sideeffects", "adverse"]):
        return "sideeffect"
    elif "benefit" in query:
        return "productbenefits"
    elif "safety" in query or "safe" in query:
        return "safetyadvice"
    elif "habit" in query or "addictive" in query:
        return "habit_forming"
    elif "chemical" in query:
        return "chemical_class"
    elif "therapeutic" in query:
        return "therapeutic_class"
    elif "action" in query:
        return "action_class"
    elif "composition" in query or "contains" in query:
        return "contains"
    elif "product" in query or "introduction" in query:
        return "productintroduction"
    else:
        return None


def search_medicine(query, info_type=None):
    matched = find_best_match(query)

    if not matched:
        return "❌ Sorry, couldn't identify that medicine."

    row = df[df["name"] == matched].iloc[0]
    name = row["name"].title()
    contains = row.get("contains", "N/A")

    # Map logical info type to actual columns
    info_map = {
        "howtouse": "howtouse",
        "sideeffect": "sideeffect",
        "productbenefits": "productbenefits",
        "safetyadvice": "safetyadvice",
        "habit_forming": "habit_forming",
        "chemical_class": "chemical_class",
        "therapeutic_class": "therapeutic_class",
        "action_class": "action_class",
        "contains": "contains",
        "productintroduction": "productintroduction"
    }

    if info_type:
        col = info_map.get(info_type)
        if col and col in df.columns:
            value = row.get(col, "")
            if value and str(value).strip():
                if col == "contains":
                    return f"🧪 **Contains of {name}**:\n{value}"
                else:
                    return f"📌 **{col.replace('_', ' ').title()} of {name}**:\n{value}"
            else:
                return "⚠️ Sorry, I couldn't find that specific information."
        else:
            return "⚠️ Sorry, I couldn't find that specific information."

    # Default full info (used for general queries)
    result = f"📘 **{name}**\n\n"
    result += f"🧪 **Composition:** {contains}\n\n"
    result += f"🔬 **Uses:** {row.get('productuses', 'N/A')}\n\n"
    result += f"📌 **Side Effects:** {row.get('sideeffect', 'N/A')}\n\n"
    result += f"🧭 **How to Use:** {row.get('howtouse', 'N/A')}\n\n"
    result += f"⚖️ **Safety Advice:** {row.get('safetyadvice', 'N/A')}\n\n"

    return result
