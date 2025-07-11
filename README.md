# 🧠 MediBot: AI-Powered Medical Chatbot

A Machine Learning and NLP-powered chatbot designed to assist users with **disease prediction** based on symptoms and provide **detailed medicine information**, including usage, side effects, composition, and more. Built with Python, Tkinter GUI, and local ML models.

---

## 📌 Table of Contents

- [🔍 Project Overview](#-project-overview)
- [⚙️ Features](#-features)
- [🖥️ Technologies Used](#-technologies-used)
- [📁 Folder Structure](#-folder-structure)
- [🚀 How It Works](#-how-it-works)
- [📸 Screenshots](#-screenshots)
- [🔧 Installation & Setup](#-installation--setup)
- [🧪 Demo Queries](#-demo-queries)
- [📚 Future Enhancements](#-future-enhancements)
- [📄 License](#-license)

---

## 🔍 Project Overview

**MediBot** is a smart desktop medical assistant that interacts with users in natural language to:
- Predict possible diseases based on symptom inputs.
- Provide medical information about drugs including **side effects**, **composition**, **benefits**, **how to use**, and **safety advice**.
- Handle general queries like greetings and thanks via intent classification.

---

## ⚙️ Features

✅ Disease prediction using symptom analysis  
✅ Medicine information (uses, composition, etc.) from a medical database  
✅ NLP-powered query understanding  
✅ Intent classification using ML  
✅ Tkinter-based GUI for clean user interaction  
✅ Local/offline functionality – no API dependency

---

## 🖥️ Technologies Used

| Technology      | Purpose                                 |
|-----------------|------------------------------------------|
| Python          | Core backend logic                      |
| Tkinter         | GUI application                        |
| SQLite          | Storing user data (optional)           |
| Pandas          | Excel file processing                  |
| scikit-learn    | ML models for intent/disease prediction |
| spaCy           | Natural language processing            |
| difflib         | Fuzzy matching for medicine search      |

---

## 📁 Folder Structure

MediBot/
│
├── app.py # Main Tkinter application
├── chatbot.py # Core chatbot logic
├── train_intent_model.py # Intent model training
│
├── helpers/
│ ├── predict_helpers.py # Disease prediction functions
│ ├── medicine_helpers.py # Medicine info functions
│ ├── nlp_helpers.py # Symptom/medicine name extraction
│ └── example_medicine_helper.py # Optional medicine alternatives
│
├── model/
│ ├── model.pkl # Trained disease prediction model
│ ├── label_encoder.pkl # Label encoder for diseases
│ ├── X_columns.pkl # List of symptom features
│ └── intent_model.pkl # Trained intent classification model
│
├── assets/
│ └── MID.xlsx # Medicine information database
│
├── data/
│ ├── Training.csv # Symptom-disease training data
│ └── description.csv # Disease description, treatments
│
└── README.md # Project documentation


---

## 🚀 How It Works

### 1. **User Input**
The user types a query like:
- "I have a headache and fever"
- "What are the side effects of Dolo 650?"

### 2. **Intent Classification**
The model detects if it's a symptom check or medicine query.

### 3. **Symptom Extraction or Medicine Match**
- Symptoms → disease model
- Medicine → Excel database lookup

### 4. **Response Generation**
A detailed response is returned to the user via the GUI.

---
MID Dataset: with 198819 rows and 16 columns
Drive link:
url : https://docs.google.com/spreadsheets/d/1mVvuAqH6g85bH5krXhf2j67JRL_PyAbZ/edit?usp=sharing&ouid=113555859551126257652&rtpof=true&sd=true

🧠 Train Intent Model:
python train_intent_model.py


▶️ Run Application:

python app.py

🧪 Demo Queries
Symptom-based

"I have a fever and cough"

"nausea, vomiting, fatigue"

Medicine-based

"What are the uses of Cetirizine?"

"Side effects of Dolo 650"

"How to use Paracetamol"

General

"Hi"

"Thank you"

"Bye"

📚 Future Enhancements
 Add voice input/output (Speech Recognition + TTS)

 API-based real-time medicine info

 Add user registration & chat history

 Improve multi-language support

📄 License
This project is for academic purposes only. For commercial use, consult the developer.


