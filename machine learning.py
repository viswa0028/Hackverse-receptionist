import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
import warnings

warnings.filterwarnings('ignore')


def train_and_save_model():
    """
    Generates synthetic data, trains a disease prediction model,
    and saves the model, label encoder, and feature selector to a pickle file.
    """
    print("Starting model training process...")

    # Extracted from your original code
    all_symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
                    'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
                    'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety',
                    'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
                    'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
                    'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin',
                    'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
                    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
                    'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
                    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
                    'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
                    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus',
                    'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
                    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
                    'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
                    'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
                    'movement_strictness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
                    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine',
                    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_typhos',
                    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
                    'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes',
                    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum',
                    'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
                    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
                    'history_of_alcohol_consumption', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations',
                    'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
                    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister',
                    'red_sore_around_nose', 'yellow_crust_ooze']
    disease_patterns = {'Fungal infection': {'primary': ['itching', 'skin_rash', 'nodal_skin_eruptions', 'scurring'],
                                             'secondary': ['dischromic_patches', 'skin_peeling', 'small_dents_in_nails',
                                                           'inflammatory_nails', 'blister'], 'probability': 0.85},
                        'Allergy': {'primary': ['continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes'],
                                    'secondary': ['runny_nose', 'congestion', 'redness_of_eyes', 'throat_irritation',
                                                  'itching'], 'probability': 0.82},
                        'GERD': {'primary': ['stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting'],
                                 'secondary': ['cough', 'chest_pain', 'throat_irritation', 'abdominal_pain'],
                                 'probability': 0.88},
                        'Chronic cholestasis': {'primary': ['itching', 'vomiting', 'yellowish_skin', 'nausea'],
                                                'secondary': ['loss_of_appetite', 'abdominal_pain', 'yellow_urine',
                                                              'yellowing_of_eyes'], 'probability': 0.9},
                        'Drug Reaction': {'primary': ['itching', 'skin_rash', 'stomach_pain', 'burning_micturition'],
                                          'secondary': ['spotting_urination', 'foul_smell_of_urine', 'fatigue',
                                                        'anxiety'], 'probability': 0.8}, 'Peptic ulcer disease': {
            'primary': ['vomiting', 'indigestion', 'loss_of_appetite', 'abdominal_pain'],
            'secondary': ['passage_of_gases', 'internal_itching', 'nausea', 'mild_fever'], 'probability': 0.87},
                        'AIDS': {
                            'primary': ['muscle_wasting', 'patches_in_throat', 'high_fever', 'extra_marital_contacts'],
                            'secondary': ['fatigue', 'weight_loss', 'swelled_lymph_nodes', 'malaise'],
                            'probability': 0.95},
                        'Diabetes': {'primary': ['fatigue', 'weight_loss', 'restlessness', 'lethargy'],
                                     'secondary': ['irregular_sugar_level', 'blurred_and_distorted_vision', 'obesity',
                                                   'excessive_hunger'], 'probability': 0.85},
                        'Gastroenteritis': {'primary': ['vomiting', 'sunken_eyes', 'dehydration', 'diarrhoea'],
                                            'secondary': ['stomach_pain', 'nausea', 'mild_fever', 'abdominal_pain'],
                                            'probability': 0.9},
                        'Bronchial Asthma': {'primary': ['fatigue', 'cough', 'high_fever', 'breathlessness'],
                                             'secondary': ['family_history', 'mucoid_sputum', 'chest_pain'],
                                             'probability': 0.83},
                        'Hypertension': {'primary': ['headache', 'chest_pain', 'dizziness', 'loss_of_balance'],
                                         'secondary': ['lack_of_concentration', 'blurred_and_distorted_vision',
                                                       'palpitations'], 'probability': 0.8},
                        'Migraine': {'primary': ['acidity', 'indigestion', 'headache', 'blurred_and_distorted_vision'],
                                     'secondary': ['excessive_hunger', 'nausea', 'depression', 'irritability'],
                                     'probability': 0.82}, 'Cervical spondylosis': {
            'primary': ['back_pain', 'weakness_in_limbs', 'neck_pain', 'dizziness'],
            'secondary': ['loss_of_balance', 'knee_pain', 'hip_joint_pain', 'muscle_weakness'], 'probability': 0.85},
                        'Paralysis (brain hemorrhage)': {
                            'primary': ['vomiting', 'headache', 'weakness_of_one_body_side', 'altered_sensorium'],
                            'secondary': ['dizziness', 'spinning_movements', 'loss_of_balance', 'unsteadiness'],
                            'probability': 0.92},
                        'Jaundice': {'primary': ['itching', 'vomiting', 'fatigue', 'weight_loss'],
                                     'secondary': ['high_fever', 'headache', 'nausea', 'loss_of_appetite',
                                                   'yellowish_skin'], 'probability': 0.88},
                        'Malaria': {'primary': ['chills', 'vomiting', 'high_fever', 'sweating'],
                                    'secondary': ['headache', 'nausea', 'diarrhoea'], 'probability': 0.9},
                        'Chicken pox': {'primary': ['itching', 'skin_rash', 'fatigue', 'lethargy'],
                                        'secondary': ['high_fever', 'headache', 'loss_of_appetite', 'mild_fever'],
                                        'probability': 0.87},
                        'Dengue': {'primary': ['skin_rash', 'chills', 'joint_pain', 'vomiting'],
                                   'secondary': ['fatigue', 'high_fever', 'headache', 'nausea', 'back_pain'],
                                   'probability': 0.9},
                        'Typhoid': {'primary': ['chills', 'vomiting', 'fatigue', 'high_fever'],
                                    'secondary': ['headache', 'nausea', 'constipation', 'abdominal_pain', 'diarrhoea'],
                                    'probability': 0.88},
                        'hepatitis A': {'primary': ['joint_pain', 'vomiting', 'yellowish_skin', 'dark_urine'],
                                        'secondary': ['nausea', 'loss_of_appetite', 'abdominal_pain', 'diarrhoea'],
                                        'probability': 0.9},
                        'Hepatitis B': {'primary': ['itching', 'fatigue', 'lethargy', 'yellowish_skin'],
                                        'secondary': ['dark_urine', 'loss_of_appetite', 'abdominal_pain',
                                                      'yellow_urine'], 'probability': 0.88},
                        'Hepatitis C': {'primary': ['fatigue', 'yellowish_skin', 'nausea', 'loss_of_appetite'],
                                        'secondary': ['yellowing_of_eyes', 'family_history', 'dark_urine'],
                                        'probability': 0.85},
                        'Hepatitis D': {'primary': ['joint_pain', 'vomiting', 'fatigue', 'yellowish_skin'],
                                        'secondary': ['dark_urine', 'nausea', 'loss_of_appetite', 'abdominal_pain'],
                                        'probability': 0.87},
                        'Hepatitis E': {'primary': ['joint_pain', 'vomiting', 'fatigue', 'yellowish_skin'],
                                        'secondary': ['dark_urine', 'nausea', 'loss_of_appetite',
                                                      'acute_liver_failure'], 'probability': 0.89},
                        'Alcoholic hepatitis': {
                            'primary': ['vomiting', 'yellowish_skin', 'abdominal_pain', 'swelling_of_stomach'],
                            'secondary': ['distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload'],
                            'probability': 0.92},
                        'Tuberculosis': {'primary': ['chills', 'vomiting', 'fatigue', 'weight_loss'],
                                         'secondary': ['cough', 'high_fever', 'breathlessness', 'sweating',
                                                       'loss_of_appetite'], 'probability': 0.9},
                        'Common Cold': {'primary': ['continuous_sneezing', 'chills', 'fatigue', 'cough'],
                                        'secondary': ['high_fever', 'headache', 'swelled_lymph_nodes', 'malaise',
                                                      'phlegm'], 'probability': 0.8},
                        'Pneumonia': {'primary': ['chills', 'fatigue', 'cough', 'high_fever'],
                                      'secondary': ['breathlessness', 'sweating', 'malaise', 'phlegm', 'chest_pain'],
                                      'probability': 0.88}, 'Dimorphic hemmorhoids(piles)': {
            'primary': ['constipation', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool'],
            'secondary': ['irritation_in_anus', 'itching', 'fatigue'], 'probability': 0.9},
                        'Heart attack': {'primary': ['vomiting', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate'],
                                         'secondary': ['breathlessness', 'sweating', 'palpitations', 'dizziness'],
                                         'probability': 0.95},
                        'Varicose veins': {'primary': ['fatigue', 'cramps', 'bruising', 'obesity'],
                                           'secondary': ['swollen_legs', 'swollen_blood_vessels',
                                                         'prominent_veins_on_calf'], 'probability': 0.83},
                        'Hypothyroidism': {'primary': ['fatigue', 'weight_gain', 'cold_hands_and_feets', 'mood_swings'],
                                           'secondary': ['lethargy', 'dizziness', 'puffy_face_and_eyes',
                                                         'enlarged_thyroid'], 'probability': 0.82},
                        'Hyperthyroidism': {'primary': ['fatigue', 'mood_swings', 'weight_loss', 'restlessness'],
                                            'secondary': ['sweating', 'diarrhoea', 'fast_heart_rate',
                                                          'excessive_hunger'], 'probability': 0.85},
                        'Hypoglycemia': {'primary': ['vomiting', 'fatigue', 'anxiety', 'sweating'],
                                         'secondary': ['headache', 'nausea', 'blurred_and_distorted_vision',
                                                       'slurred_speech'], 'probability': 0.8},
                        'Osteoarthristis': {'primary': ['joint_pain', 'neck_pain', 'knee_pain', 'hip_joint_pain'],
                                            'secondary': ['muscle_weakness', 'stiff_neck', 'swelling_joints',
                                                          'movement_strictness'], 'probability': 0.85}, 'Arthritis': {
            'primary': ['muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_strictness'],
            'secondary': ['joint_pain', 'knee_pain', 'hip_joint_pain', 'painful_walking'], 'probability': 0.83},
                        '(vertigo) Paroymsal  Positional Vertigo': {
                            'primary': ['vomiting', 'headache', 'nausea', 'spinning_movements'],
                            'secondary': ['loss_of_balance', 'unsteadiness', 'dizziness'], 'probability': 0.88},
                        'Acne': {'primary': ['skin_rash', 'pus_filled_pimples', 'blackheads', 'scurring'],
                                 'secondary': ['skin_peeling', 'dischromic_patches', 'inflammatory_nails'],
                                 'probability': 0.85}, 'Urinary tract infection': {
            'primary': ['burning_micturition', 'foul_smell_of_urine', 'continuous_feel_of_urine'],
            'secondary': ['bladder_discomfort', 'spotting_urination', 'abdominal_pain'], 'probability': 0.9},
                        'Psoriasis': {'primary': ['skin_rash', 'joint_pain', 'skin_peeling', 'silver_like_dusting'],
                                      'secondary': ['small_dents_in_nails', 'inflammatory_nails', 'blister'],
                                      'probability': 0.87},
                        'Impetigo': {'primary': ['skin_rash', 'high_fever', 'blister', 'red_sore_around_nose'],
                                     'secondary': ['yellow_crust_ooze', 'itching', 'dischromic_patches'],
                                     'probability': 0.85}}

    # 2. GENERATE SYNTHETIC DATA
    print("Generating synthetic training data...")
    np.random.seed(42)
    synthetic_data = []
    for disease, pattern in disease_patterns.items():
        n_samples = np.random.randint(70, 101)
        for _ in range(n_samples):
            sample = {symptom: 0 for symptom in all_symptoms}
            for symptom in pattern['primary']:
                if symptom in sample and np.random.random() < pattern['probability']:
                    sample[symptom] = 1
            for symptom in pattern['secondary']:
                if symptom in sample and np.random.random() < (pattern['probability'] * 0.65):
                    sample[symptom] = 1
            for symptom in all_symptoms:
                if sample[symptom] == 0 and np.random.random() < 0.015:
                    sample[symptom] = 1
            sample['prognosis'] = disease
            synthetic_data.append(sample)

    train_df = pd.DataFrame(synthetic_data)

    # 3. PREPARE DATA FOR TRAINING
    print("Preparing data for training...")
    X = train_df[all_symptoms].values
    y = train_df['prognosis'].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    feature_selector = SelectKBest(score_func=chi2, k=min(35, len(all_symptoms)))
    X_selected = feature_selector.fit_transform(X, y_encoded)
    X_train, X_val, y_train, y_val = train_test_split(
        X_selected, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )

    # 4. TRAIN MODELS AND SELECT THE BEST ONE
    print("Training multiple models...")
    models = {
        'rf': RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=3, min_samples_leaf=2,
                                     random_state=42, class_weight='balanced'),
        'gb': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=8, random_state=42),
        'lr': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1500, solver='liblinear'),
        'nb': MultinomialNB(alpha=0.3)
    }
    trained_models = []
    best_model = None
    best_score = 0
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            score = accuracy_score(y_val, model.predict(X_val))
            print(f"  - {name} model accuracy: {score:.4f}")
            trained_models.append((name, model))
            if score > best_score:
                best_score = score
                best_model = model
        except Exception as e:
            print(f"Could not train {name}: {e}")

    # Create and evaluate an ensemble
    if len(trained_models) >= 3:
        print("Training ensemble model...")
        ensemble = VotingClassifier(estimators=trained_models[:3], voting='soft')
        ensemble.fit(X_train, y_train)
        ensemble_score = accuracy_score(y_val, ensemble.predict(X_val))
        print(f"  - Ensemble model accuracy: {ensemble_score:.4f}")
        if ensemble_score > best_score:
            best_model = ensemble
            best_score = ensemble_score

    print(f"\nBest performing model selected with accuracy: {best_score:.4f}")

    # 5. SAVE THE COMPONENTS TO A PICKLE FILE
    model_components = {
        'model': best_model,
        'label_encoder': label_encoder,
        'feature_selector': feature_selector,
        'all_symptoms_list': all_symptoms
    }

    with open('disease_model.pkl', 'wb') as f:
        pickle.dump(model_components, f)

    print("\n✅ Model components saved successfully to 'disease_model.pkl'")


if __name__ == "__main__":
    train_and_save_model()