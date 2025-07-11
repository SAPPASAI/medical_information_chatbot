import pandas as pd
from fuzzywuzzy import fuzz

# Load your medicine dataset
med_df = pd.read_csv("data/medicine_with_real_prices.csv")  # Ensure correct path and filename

def extract_price(price_str):
    try:
        if isinstance(price_str, str):
            return float(price_str.lower().replace("rs", "").strip())
        elif isinstance(price_str, (int, float)):
            return float(price_str)
        else:
            return float('inf')
    except:
        return float('inf')  # Handle missing/invalid prices by pushing them to the end

def find_alternative_medicines(medicine_name, top_n=5):
    medicine_name = medicine_name.lower().strip()
    matches = []

    for idx, row in med_df.iterrows():
        try:
            med_name = str(row.get('Drug_Name', '')).lower()
            price = extract_price(row.get('Price', ''))
            score = fuzz.token_sort_ratio(medicine_name, med_name)

            if score > 60:
                matches.append({
                    "name": row.get('Drug_Name', 'Unknown'),
                    "score": score,
                    "price": price,
                    "desc": row.get("Description", "No description available"),
                    "reason": row.get("Reason", "Unknown reason")
                })
        except Exception as e:
            continue

    # Sort first by descending similarity score, then ascending price
    matches = sorted(matches, key=lambda x: (-x["score"], x["price"]))

    # Skip exact match and get top_n alternatives
    filtered = []
    for match in matches:
    # Exclude exact or near-exact matches (similarity >= 95)
          if fuzz.token_sort_ratio(medicine_name, match["name"].lower()) < 95:
               filtered.append(match)
          if len(filtered) == top_n:
               break


    # Format the response lines
    result_lines = []
    for med in filtered:
        display_price = f"{med['price']} rs" if med['price'] != float('inf') else "Price not available"
        result_lines.append(f"💊 {med['name']} - {display_price}\n📝 {med['desc']}\n")

    return result_lines
