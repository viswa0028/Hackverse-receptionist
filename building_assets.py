import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE

# Import the knowledge graph class from the other file
from knowledge_graph import MedicalKnowledgeGraph


def create_and_save_assets():
    """
    Trains all ML models, builds the knowledge graph, and saves them
    into a single file for the main application to use.
    """
    print("🚀 STARTING ASSET BUILDING PROCESS 🚀")

    # --- Step 1: Define Knowledge Base ---
    comprehensive_disease_patterns = {
        'Fungal infection': {'primary': ['itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic _patches'],
                             'secondary': ['scurring', 'skin_peeling'], 'probability': 0.85},
        'Allergy': {'primary': ['continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes'],
                    'secondary': ['runny_nose', 'congestion'], 'probability': 0.8},
        # ... (include all 42 disease patterns here) ...
        'Impetigo': {'primary': ['skin_rash', 'high_fever', 'blister', 'red_sore_around_nose'],
                     'secondary': ['yellow_crust_ooze', 'swelled_lymph_nodes'], 'probability': 0.85},
    }

    # --- Step 2: Build and Prepare Knowledge Graph ---
    knowledge_graph = MedicalKnowledgeGraph()
    knowledge_graph.build_from_patterns(comprehensive_disease_patterns)

    # --- Step 3: Generate or Load Training Data ---
    # Using the same data generation logic from your script
    symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
                'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
                'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety',
                'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat',
                'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
                'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite',
                'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever',
                'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
                'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
                'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
                'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
                'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
                'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
                'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
                'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
                'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
                'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
                'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression',
                'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
                'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
                'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
                'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
                'distention_of_abdomen', 'history_of_alcohol_consumption', 'blood_in_sputum', 'prominent_veins_on_calf',
                'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
                'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
                'yellow_crust_ooze']
    diseases = list(comprehensive_disease_patterns.keys())
    train_samples = []
    for disease in diseases:
        for _ in range(20):
            sample = {s: 0 for s in symptoms}
            n_symptoms = np.random.randint(3, 7)
            selected_symptoms = np.random.choice(symptoms, size=n_symptoms, replace=False)
            for symptom in selected_symptoms:
                sample[symptom] = 1
            sample['prognosis'] = disease
            train_samples.append(sample)
    train_df = pd.DataFrame(train_samples)

    # --- Step 4: Train the ML Model ---
    symptom_columns = [col for col in train_df.columns if col != 'prognosis']
    X = train_df[symptom_columns].values
    y = train_df['prognosis'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    feature_selector = SelectKBest(score_func=chi2, k=min(70, len(symptom_columns) // 2))
    X_selected = feature_selector.fit_transform(X, y_encoded)

    X_train, X_val, y_train, y_val = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42,
                                                      stratify=y_encoded)

    smote = SMOTE(random_state=42, k_neighbors=min(5, len(np.unique(y_train)) - 1))
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    best_model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
    print("Training the best model (Random Forest)...")
    best_model.fit(X_train_smote, y_train_smote)
    print("Model training complete.")

    # --- Step 5: Save All Assets to a Single File ---
    print("\n💾 Saving all assets to 'medical_assets.joblib'...")
    assets = {
        'ml_model': best_model,
        'label_encoder': label_encoder,
        'feature_selector': feature_selector,
        'symptom_columns': symptom_columns,
        'knowledge_graph': knowledge_graph
    }
    joblib.dump(assets, 'medical_assets.joblib')
    print("✅ Assets saved successfully!")


if __name__ == "__main__":
    create_and_save_assets()