import pandas as pd

# Load the provided MID.xlsx to inspect column structure and sample values
file_path = "E:/trail/backend3/backend/assets/MID.xlsx"
df = pd.read_excel(file_path)

# Normalize columns
df.columns = df.columns.str.strip().str.lower()

# Show all column names and a few sample rows
df.columns.tolist(), df.head(3)
