import psycopg2

# --- CHOOSE YOUR CONNECTION ---
# Option 1: For your LOCAL database
db_params = {
    "host": "localhost",
    "port": "5432",
    "dbname": "hospital_receptionist_db",
    "user": "postgres",
    "password": "Visu@2006"  # <-- Replace with your local password
}


def add_new_patient(first_name, last_name, dob, phone, email):
    """Adds a new patient to the Patients table."""
    # SQL query to insert a new patient
    sql = """
        INSERT INTO Patients (first_name, last_name, date_of_birth, phone_number, email)
        VALUES (%s, %s, %s, %s, %s) RETURNING patient_id;
    """

    conn = None
    patient_id = None

    try:
        # Connect to the database
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # Execute the command with the patient's data
        cursor.execute(sql, (first_name, last_name, dob, phone, email))

        # Get the generated patient_id back
        patient_id = cursor.fetchone()[0]

        # Commit the changes to the database
        conn.commit()

        print(f"✅ Patient '{first_name} {last_name}' added successfully with ID: {patient_id}")

        # Close the cursor
        cursor.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"❌ Error while adding patient: {error}")
    finally:
        if conn is not None:
            conn.close()


# --- Example of how to use the function ---
if __name__ == '__main__':
    add_new_patient(
        first_name="Sita",
        last_name="Gupta",
        dob="1985-11-20",
        phone="+919123456789",
        email="sita.gupta@example.com"
    )