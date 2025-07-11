import sqlite3

conn = sqlite3.connect("medical_chatbot.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS User (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS ChatHistory (
    chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    message_text TEXT NOT NULL,
    sender_type TEXT CHECK(sender_type IN ('User', 'Bot')) NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES User(user_id)
);
""")

cursor.execute('''
CREATE TABLE IF NOT EXISTS Prediction (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    symptoms TEXT,
    predicted_disease TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')





cursor.execute("""
CREATE TABLE IF NOT EXISTS MedicineInfo (
    medicine_id INTEGER PRIMARY KEY AUTOINCREMENT,
    medicine_name TEXT,
    uses TEXT,
    side_effects TEXT,
    precautions TEXT
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS AlternativeMedicine (
    alt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    medicine_id INTEGER,
    alternative_name TEXT,
    FOREIGN KEY (medicine_id) REFERENCES MedicineInfo(medicine_id)
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS Feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    message_text TEXT NOT NULL,
    feedback TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES User(user_id)
);
""")

conn.commit()
conn.close()
