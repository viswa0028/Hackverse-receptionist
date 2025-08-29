import os
import psycopg2
from dotenv import load_dotenv


def add_medical_specializations():
    """
    Connects to the database and bulk-inserts a list of 50
    common medical specializations.
    """
    print("--- 🚀 Starting to Add Medical Specializations ---")
    load_dotenv()

    specializations_to_add = [
        'Allergy and immunology', 'Anesthesiology', 'Dermatology', 'Diagnostic radiology',
        'Emergency medicine', 'Family medicine', 'Internal medicine', 'Medical genetics',
        'Neurology', 'Nuclear medicine', 'Obstetrics and gynecology', 'Ophthalmology',
        'Pathology', 'Pediatrics', 'Physical medicine and rehabilitation', 'Preventive medicine',
        'Psychiatry', 'Radiation oncology', 'Surgery', 'Urology', 'Cardiology',
        'Endocrinology', 'Gastroenterology', 'Hematology', 'Infectious disease',
        'Nephrology', 'Oncology', 'Pulmonology', 'Rheumatology', 'Geriatrics',
        'Hospice and palliative medicine', 'Sleep medicine', 'Sports medicine',
        'Otolaryngology', 'Plastic surgery', 'Thoracic surgery', 'Vascular surgery',
        'Orthopedics', 'Anesthesiologist', 'Cardiologist', 'Dermatologist', 'Endocrinologist',
        'Gastroenterologist', 'Hematologist', 'Immunologist', 'Neurologist', 'Oncologist',
        'Orthopedist', 'Pediatrician', 'Psychiatrist', 'Radiologist', 'Rheumatologist',
        'Surgeon', 'Urologist', 'General Physician'
    ]

    conn = None
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS")
        )
        cursor = conn.cursor()
        print("✅ Database connection successful.")

        insert_count = 0
        for spec_name in specializations_to_add:
            cursor.execute(
                "INSERT INTO Specializations (specialization_name) VALUES (%s) ON CONFLICT (specialization_name) DO NOTHING;",
                (spec_name,)
            )
            # cursor.rowcount will be 1 if a row was inserted, 0 otherwise
            insert_count += cursor.rowcount

        conn.commit()
        print(f"✅ Operation complete. Added {insert_count} new specializations.")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"❌ An error occurred: {error}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("--- Database connection closed. ---")


if __name__ == "__main__":
    add_medical_specializations()