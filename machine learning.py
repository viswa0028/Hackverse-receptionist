import joblib
import numpy as np
from typing import List
# The LLM Arbiter now also becomes part of the main app
class LLMArbiterLayer:
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph

    def decide_prediction(self, symptoms: List[str], ml_prediction: str, kg_prediction: str):
        ml_patterns = self.knowledge_graph.comprehensive_disease_patterns.get(ml_prediction, {})
        kg_patterns = self.knowledge_graph.comprehensive_disease_patterns.get(kg_prediction, {})
        ml_primary_symptoms = set(ml_patterns.get('primary', []))
        kg_primary_symptoms = set(kg_patterns.get('primary', []))
        normalized_input_symptoms = {self.knowledge_graph._normalize_symptom_name(s) for s in symptoms}
        ml_primary_match_count = len(normalized_input_symptoms.intersection(ml_primary_symptoms))
        kg_primary_match_count = len(normalized_input_symptoms.intersection(kg_primary_symptoms))

        if ml_primary_match_count > kg_primary_match_count:
            final_decision, reason = ml_prediction, f"Symptoms align better with '{ml_prediction}'."
        elif kg_primary_match_count > ml_primary_match_count:
            final_decision, reason = kg_prediction, f"Symptoms align better with '{kg_prediction}'."
        else:
            final_decision, reason = ml_prediction, "Evidence is ambiguous. Defaulting to ML model."
        return {"final_decision": final_decision, "reasoning": reason}


class PredictionSystem:
    def __init__(self, assets_path='medical_assets.joblib'):
        print("Loading medical assets...")
        assets = joblib.load(assets_path)
        self.ml_model = assets['ml_model']
        self.label_encoder = assets['label_encoder']
        self.feature_selector = assets['feature_selector']
        self.symptom_columns = assets['symptom_columns']
        self.knowledge_graph = assets['knowledge_graph']
        self.llm_arbiter = LLMArbiterLayer(self.knowledge_graph)
        print("✅ System ready.")

    def predict(self, symptoms_list: List[str]):
        # 1. ML Prediction
        feature_vector = np.zeros(len(self.symptom_columns))
        for symptom in symptoms_list:
            symptom_clean = self.knowledge_graph._normalize_symptom_name(symptom)
            if symptom_clean in self.symptom_columns:
                idx = self.symptom_columns.index(symptom_clean)
                feature_vector[idx] = 1

        feature_vector_selected = self.feature_selector.transform([feature_vector])
        prediction_encoded = self.ml_model.predict(feature_vector_selected)[0]
        ml_disease = self.label_encoder.inverse_transform([prediction_encoded])[0]
        ml_confidence = np.max(self.ml_model.predict_proba(feature_vector_selected))

        # 2. KG Validation
        kg_validation = self.knowledge_graph.validate_prediction(symptoms_list, ml_disease, ml_confidence)

        # 3. LLM Arbitration if needed
        if kg_validation['is_consistent']:
            print("✅ ML and KG are consistent.")
            return {"source": "ML Model (Validated by KG)", "prediction": ml_disease}
        else:
            print("⚠️ ML and KG have differing views. Invoking LLM Arbiter.")
            kg_suggestion = kg_validation['alternative_diseases'][0]['disease']
            arbitration = self.llm_arbiter.decide_prediction(symptoms_list, ml_disease, kg_suggestion)
            return {"source": "LLM Arbiter", "prediction": arbitration['final_decision'],
                    "details": arbitration['reasoning']}

if __name__ == "__main__":
    system = PredictionSystem()

    # 2. Define a test case
    test_symptoms = [" Heartbeat","sweating","pain in chest"]

    # 3. Get a prediction
    final_prediction = system.predict(test_symptoms)

    print("\n--- FINAL DIAGNOSTIC RESULT ---")
    print(f"Source of Decision: {final_prediction['source']}")
    print(f"Final Prediction: {final_prediction['prediction']}")
    print(f"Justification: {final_prediction.get('details', 'N/A')}")
    print("-----------------------------")