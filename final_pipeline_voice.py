import os
import re
import json
import pickle
import numpy as np
import psycopg2
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime, timedelta
from googletrans import Translator
import pyttsx3
import whisper  # New import
import sounddevice as sd  # New import
import queue
import threading

print("--- 🚀 Initializing Whisper-Powered Voice/Text Pipeline ---")
load_dotenv()
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    print("✅ Gemini configured successfully.")
except Exception as e:
    print(f"❌ Failed to configure Gemini: {e}")
    exit()
try:
    with open('disease_model.pkl', 'rb') as f:
        model_components = pickle.load(f)
    ml_model, label_encoder, feature_selector, all_symptoms = model_components['model'], model_components[
        'label_encoder'], model_components['feature_selector'], model_components['all_symptoms_list']
    print("✅ ML model 'disease_model.pkl' loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'disease_model.pkl' not found. Please run train_model.py first.")
    exit()
translator = Translator()
tts_engine = pyttsx3.init()

# Load the Whisper model
try:
    whisper_model = whisper.load_model("base")  # "base" is a small and fast model
    print("✅ Whisper model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load Whisper model: {e}")
    exit()


# --- 2. CORE FUNCTIONS (New and Existing) ---

def speak_text(text, lang='en'):
    # (This function is unchanged)
    try:
        if lang == 'hi':
            voices = tts_engine.getProperty('voices')
            hindi_voice = next((voice for voice in voices if "hindi" in voice.name.lower() or "hi" in voice.lang), None)
            if hindi_voice:
                tts_engine.setProperty('voice', hindi_voice.id)
            else:
                print("(Hindi voice not found, using default)")
        else:
            voices = tts_engine.getProperty('voices')
            english_voice = next((voice for voice in voices if "english" in voice.name.lower()), voices[0])
            tts_engine.setProperty('voice', english_voice.id)
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"❌ Error during text-to-speech: {e}")


# --- NEW WHISPER FUNCTION FOR SPEECH INPUT ---
def get_speech_input_whisper() -> str:
    """Uses OpenAI's Whisper to capture and transcribe speech from the microphone."""
    samplerate = 16000  # Whisper works with 16kHz audio
    q = queue.Queue()
    is_listening = threading.Event()
    is_listening.set()

    def callback(indata, frames, time, status):
        """This is called for each audio block from the microphone."""
        if status:
            print(status)
        q.put(indata.copy())

    print("\n🎤 Listening... Please state your symptoms (Recording for 10 seconds).")
    speak_text("Listening... Please state your symptoms now.", lang='en')

    try:
        with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, dtype='float32'):
            # Record for a fixed duration (e.g., 10 seconds)
            recording = []
            for _ in range(int(samplerate / 1024 * 10)):  # 10 seconds of recording
                recording.append(q.get())
        print("...Transcribing audio with Whisper...")
        audio_data = np.concatenate(recording, axis=0).flatten()

        result = whisper.transcribe(whisper_model, audio_data)
        return result['text'].strip()

    except Exception as e:
        print(f"An error occurred with Whisper: {e}")
        return None


# ... (All your other core functions: get_db_connection, normalize_symptoms, etc., remain the same) ...
def get_db_connection():
    try:
        return psycopg2.connect(host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"), dbname=os.getenv("DB_NAME"),
                                user=os.getenv("DB_USER"), password=os.getenv("DB_PASS"))
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None


def normalize_symptoms(user_input: str) -> list:
    symptom_list_str = "'" + "', '".join(all_symptoms) + "'"
    prompt = f"""You are an expert medical symptom normalizer. ### INSTRUCTIONS: - Your task is to extract all current, valid symptoms from the "User Input". - You MUST map each extracted symptom to the closest matching symptom from the "Symptom List". - You MUST ignore any symptoms described with negations (e.g., "no fever", "cough is gone"). - Your output MUST be a valid JSON array of strings and nothing else. If no valid symptoms are found, return an empty array []. ### SYMPTOM LIST: [{symptom_list_str}] ### EXAMPLES: User Input: "Hi, my head has been hurting a lot and I feel sick to my stomach. My fever from last week is gone." Output: ["headache", "nausea"] ### USER INPUT: {user_input} ### OUTPUT:"""
    try:
        response = gemini_model.generate_content(prompt)
        cleaned_text = re.sub(r"```json|```", "", response.text.strip())
        return json.loads(cleaned_text)
    except Exception as e:
        print(f"Error during symptom normalization: {e}")
        return []


def predict_disease(symptoms_list: list) -> str:
    input_vector = np.zeros(len(all_symptoms))
    for symptom in symptoms_list:
        if symptom in all_symptoms:
            input_vector[all_symptoms.index(symptom)] = 1
    input_selected = feature_selector.transform([input_vector])
    prediction_encoded = ml_model.predict(input_selected)[0]
    return label_encoder.inverse_transform([prediction_encoded])[0]


def find_specialist_and_doctors(disease: str, conn):
    prompt = f"Given a medical condition, provide only the corresponding specialist's name. Condition: Diabetes Specialist: Endocrinologist Condition: {disease} Specialist:"
    try:
        response = gemini_model.generate_content(prompt)
        specialist_name = response.text.strip()
    except Exception:
        specialist_name = "General Physician"
    print(f"  -> Identified specialist: {specialist_name}")
    print(f"  -> Searching database for doctors...")
    cursor = conn.cursor()
    cursor.execute(
        """SELECT d.doctor_id, d.first_name, d.last_name, h.name AS hospital_name, array_agg(sch.day_of_week), array_agg(sch.start_time), array_agg(sch.end_time) FROM Doctors d JOIN Hospitals h ON d.hospital_id = h.hospital_id JOIN Doctor_Specializations ds ON d.doctor_id = ds.doctor_id JOIN Specializations s ON ds.specialization_id = s.specialization_id LEFT JOIN Doctor_Schedules sch ON d.doctor_id = sch.doctor_id WHERE s.specialization_name = %s GROUP BY d.doctor_id, d.first_name, d.last_name, h.name;""",
        (specialist_name,))
    doctors = cursor.fetchall()
    cursor.close()
    return specialist_name, doctors


def generate_available_slots(doctors: list, conn, appointment_duration_mins=30):
    available_slots = {}
    today = datetime.now()
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    doctor_ids = [doc[0] for doc in doctors]
    booked_slots_set = set()
    if doctor_ids:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT doctor_id, appointment_time FROM Appointments WHERE doctor_id = ANY(%s) AND appointment_time >= %s AND status = 'Booked';",
            (doctor_ids, today))
        booked_appointments = cursor.fetchall()
        for doc_id, appt_time in booked_appointments:
            booked_slots_set.add((doc_id, appt_time.replace(tzinfo=None)))
        cursor.close()
    for doc in doctors:
        doc_id, first_name, last_name, hospital, work_days, start_times, end_times = doc
        doc_key = (doc_id, f"Dr. {first_name} {last_name} ({hospital})")
        available_slots[doc_key] = []
        if not work_days or not work_days[0]: continue
        schedule_map = {day: (start, end) for day, start, end in zip(work_days, start_times, end_times)}
        for i in range(7):
            current_day = today + timedelta(days=i)
            current_day_name = days_of_week[current_day.weekday()]
            if current_day_name in schedule_map:
                start_time, end_time = schedule_map[current_day_name]
                slot_time = datetime.combine(current_day.date(), start_time)
                end_slot_time = datetime.combine(current_day.date(), end_time)
                while slot_time + timedelta(minutes=appointment_duration_mins) <= end_slot_time:
                    if (doc_id, slot_time) not in booked_slots_set:
                        available_slots[doc_key].append(slot_time)
                    slot_time += timedelta(minutes=appointment_duration_mins)
    return available_slots


# --- 4. MAIN EXECUTION PIPELINE with INTERACTIVE CHOICE ---

if __name__ == "__main__":

    # --- Step 1: Ask user for input method ---
    user_input = None
    while not user_input:
        choice = input("How would you like to provide your symptoms? (Enter 'text' or 'speech'): ").lower().strip()

        if choice == 'text':
            user_input = input("Please type your symptoms in English or Hindi:\n")
        elif choice == 'speech':
            user_input = get_speech_input_whisper()  # Using the new Whisper function
            if user_input:
                print(f"Whisper heard: \"{user_input}\"")
        else:
            print("Invalid choice. Please enter 'text' or 'speech'.")

    if not user_input:
        print("No input received. Exiting.")
        exit()

    # --- Step 2: Detect Language, Translate, and Speak Welcome ---
    # (The rest of the pipeline is the same as before)
    try:
        detected_lang = translator.detect(user_input).lang
        if detected_lang == 'hi':
            english_input = translator.translate(user_input, dest='en').text
            welcome_message = "नमस्ते, मैं आपके लक्षणों का विश्लेषण कर रहा हूँ।"
        else:
            english_input = user_input
            welcome_message = "Hello, I am analyzing your symptoms."

        print(f"\n[ANALYSIS] {welcome_message}")
        speak_text(welcome_message, lang=detected_lang)
    except Exception as e:
        print(f"Translation/Detection failed: {e}. Defaulting to English.")
        english_input = user_input
        detected_lang = 'en'

    # --- Step 3: Run the Core English-Based Pipeline ---
    normalized_symptoms = normalize_symptoms(english_input)
    if not normalized_symptoms:
        error_msg = "Could not identify valid symptoms."
        if detected_lang == 'hi': error_msg = translator.translate(error_msg, dest='hi').text
        print(error_msg)
        speak_text(error_msg, lang=detected_lang)
        exit()

    predicted_disease = predict_disease(normalized_symptoms)
    db_conn = get_db_connection()
    if not db_conn: exit()
    specialist, doctors = find_specialist_and_doctors(predicted_disease, db_conn)
    slots = generate_available_slots(doctors, db_conn)
    db_conn.close()

    # --- Step 4: Format, Translate, and Speak Final Output ---
    print("\n--- ✅ Final Results ---")
    final_output_lines = []
    if not any(slots.values()):
        final_output_lines.append("Sorry, no available appointment slots were found.")
    else:
        header = f"Based on the prediction of {predicted_disease}, here are the available slots for a {specialist}:"
        final_output_lines.append(header)
        for (doc_id, doc_name), doc_slots in slots.items():
            if doc_slots:
                final_output_lines.append(f"\nFor {doc_name} with ID {doc_id}:")
                for slot_datetime in doc_slots[:3]:
                    final_output_lines.append(slot_datetime.strftime('%A at %I:%M %p.'))

    for line in final_output_lines:
        if detected_lang == 'hi':
            translated_line = translator.translate(line, dest='hi').text
            print(translated_line)
            speak_text(translated_line, lang='hi')
        else:
            print(line)
            speak_text(line, lang='en')