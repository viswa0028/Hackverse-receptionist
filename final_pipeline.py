import os
import re
import json
import pickle
import numpy as np
import psycopg2
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. INITIAL SETUP AND LOADING ---
print("--- 🚀 Initializing Full Pipeline ---")

# Load environment variables from .env file
load_dotenv()

# Configure Gemini
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel("gemini-2.5-pro")
    print("✅ Gemini configured successfully.")
except Exception as e:
    print(f"❌ Failed to configure Gemini: {e}")
    exit()

# Load the trained ML model and its components
try:
    with open('disease_model.pkl', 'rb') as f:
        model_components = pickle.load(f)
    ml_model = model_components['model']
    label_encoder = model_components['label_encoder']
    feature_selector = model_components['feature_selector']
    all_symptoms = model_components['all_symptoms_list']
    print("✅ ML model 'disease_model.pkl' loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'disease_model.pkl' not found. Please run train_model.py first.")
    exit()


# --- 2. CORE FUNCTIONS (from your 3 separate files) ---

def extract_symptoms_from_sentence(user_input: str) -> list:
    """
    (From test_symptom_extractor.py)
    Uses Gemini to extract and normalize symptoms from a sentence.
    """
    symptom_list_str = "'" + "', '".join(all_symptoms) + "'"
    prompt = f"""
    You are an expert medical symptom normalizer.
    ### INSTRUCTIONS:
    - Your task is to extract all current, valid symptoms from the "User Input".
    - You MUST map each extracted symptom to the closest matching symptom from the "Symptom List".
    - You MUST ignore any symptoms described with negations (e.g., "no fever", "cough is gone").
    - Your output MUST be a valid JSON array of strings and nothing else.
    ### SYMPTOM LIST:
    [{symptom_list_str}]
    ### USER INPUT:
    {user_input}
    ### OUTPUT:
    """
    try:
        response = gemini_model.generate_content(prompt)
        cleaned_text = re.sub(r"```json|```", "", response.text.strip())
        return json.loads(cleaned_text)
    except Exception as e:
        print(f"An error occurred during symptom extraction: {e}")
        return []


def predict_disease_from_symptoms(symptoms_list: list) -> str:
    """
    (From test_ml_predictor.py)
    Predicts a disease from a list of normalized symptoms.
    """
    try:
        input_vector = np.zeros(len(all_symptoms))
        for symptom in symptoms_list:
            if symptom in all_symptoms:
                input_vector[all_symptoms.index(symptom)] = 1

        input_selected = feature_selector.transform([input_vector])
        prediction_encoded = ml_model.predict(input_selected)[0]
        return label_encoder.inverse_transform([prediction_encoded])[0]
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return "Prediction Failed"


def find_doctors_from_database(specialty_name: str) -> list:
    """
    (From test_doctor_finder.py)
    Connects to the database and finds doctors for a specific specialty.
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS")
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT d.first_name, d.last_name, h.name AS hospital_name
            FROM Doctors d
            JOIN Hospitals h ON d.hospital_id = h.hospital_id
            JOIN Doctor_Specializations ds ON d.doctor_id = ds.doctor_id
            JOIN Specializations s ON ds.specialization_id = s.specialization_id
            WHERE s.specialization_name = %s;
        """, (specialty_name,))
        return cursor.fetchall()
    except Exception as e:
        print(f"An error occurred during database query: {e}")
        return []
    finally:
        if conn:
            conn.close()


def get_specialist_for_disease(disease: str) -> str:
    """Uses Gemini to find the appropriate specialist for a disease."""
    prompt = f"Given a medical condition, provide only the corresponding specialist's name. Condition: Diabetes Specialist: Endocrinologist Condition: {disease} Specialist:"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "General Physician"


# --- 3. MAIN EXECUTION BLOCK ---

if __name__ == "__main__":

    # Define the user's input sentence
    user_input_sentence = "Hi, I've been having a really tight chest pain that sometimes goes to my left arm, and I feel short of breath and a bit dizzy. My heart is beating really fast too."

    print(f"\n[INPUT] User says: \"{user_input_sentence}\"")
    print("-" * 50)

    # --- Step 1: Extract Symptoms ---
    print("[STEP 1] Extracting symptoms from sentence...")
    normalized_symptoms = extract_symptoms_from_sentence(user_input_sentence)
    if not normalized_symptoms:
        print("  -> Could not identify valid symptoms. Exiting.")
        exit()
    print(f"  -> Identified Symptoms: {normalized_symptoms}")
    print("-" * 50)

    # --- Step 2: Predict Disease ---
    print("[STEP 2] Predicting disease from symptoms...")
    predicted_disease = predict_disease_from_symptoms(normalized_symptoms)
    print(f"  -> Predicted Condition: {predicted_disease}")
    print("-" * 50)

    # --- Step 3: Find Specialist and Doctors ---
    print("[STEP 3] Finding relevant doctors from database...")
    specialist = get_specialist_for_disease(predicted_disease)
    print(f"  -> Required Specialist: {specialist}")

    available_doctors = find_doctors_from_database(specialist)

    # --- Final Output ---
    print("\n--- ✅ PIPELINE COMPLETE: RESULTS ---")
    if available_doctors:
        print(f"Found the following doctors for '{specialist}':")
        for doc in available_doctors:
            print(f"  - Dr. {doc[0]} {doc[1]} at {doc[2]}")
    else:
        print(f"⚠️ No doctors found for the specialty '{specialist}' in the database.")