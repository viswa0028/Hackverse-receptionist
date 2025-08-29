import networkx as nx
import re
from typing import Dict, List, Tuple


class MedicalKnowledgeGraph:
    """Medical knowledge graph for validating disease predictions"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.symptom_to_disease = {}
        self.disease_to_symptoms = {}
        self.comprehensive_disease_patterns = {}

    def build_from_patterns(self, disease_patterns: Dict) -> None:
        """Build knowledge graph from disease patterns"""
        self.comprehensive_disease_patterns = disease_patterns
        for disease, pattern in disease_patterns.items():
            self.graph.add_node(disease, type='disease')
            for symptom in pattern['primary']:
                self._add_symptom_relationship(symptom, disease, 'primary', pattern['probability'])
            for symptom in pattern['secondary']:
                self._add_symptom_relationship(symptom, disease, 'secondary', pattern['probability'] * 0.7)

    def _add_symptom_relationship(self, symptom: str, disease: str, relationship_type: str, weight: float) -> None:
        """Add a symptom-disease relationship to the graph"""
        if symptom not in self.graph:
            self.graph.add_node(symptom, type='symptom')
        self.graph.add_edge(symptom, disease, relationship=relationship_type, weight=weight)
        if symptom not in self.symptom_to_disease:
            self.symptom_to_disease[symptom] = {}
        self.symptom_to_disease[symptom][disease] = weight
        if disease not in self.disease_to_symptoms:
            self.disease_to_symptoms[disease] = {}
        self.disease_to_symptoms[disease][symptom] = weight

    # ... (All other methods from your MedicalKnowledgeGraph class remain here) ...
    # _normalize_symptom_name, validate_prediction, _calculate_kg_confidence, etc.
    def _normalize_symptom_name(self, symptom: str) -> str:
        normalized = symptom.lower().strip().replace(' ', '_')
        normalized = re.sub(r'[^\w_]', '', normalized)
        normalized = re.sub(r'_+', '_', normalized)
        return normalized

    def validate_prediction(self, symptoms: List[str], predicted_disease: str, confidence: float,
                            threshold: float = 0.3) -> Dict:
        normalized_symptoms = [self._normalize_symptom_name(s) for s in symptoms]
        kg_confidence, matched_symptoms = self._calculate_kg_confidence(normalized_symptoms, predicted_disease)
        is_consistent = kg_confidence >= threshold
        alternatives = self._get_alternative_diseases(normalized_symptoms, predicted_disease, top_n=3)
        symptom_explanations = self._explain_symptom_relevance(normalized_symptoms, predicted_disease)
        return {'is_consistent': is_consistent, 'kg_confidence': kg_confidence, 'ml_confidence': confidence,
                'matched_symptoms': matched_symptoms, 'alternative_diseases': alternatives,
                'symptom_explanations': symptom_explanations, 'validation_passed': is_consistent or confidence > 0.7}

    def _calculate_kg_confidence(self, symptoms: List[str], disease: str) -> Tuple[float, List[str]]:
        if disease not in self.disease_to_symptoms: return 0.0, []
        disease_symptoms = set(self.disease_to_symptoms[disease].keys())
        input_symptoms = set(symptoms)
        matched_symptoms = list(disease_symptoms.intersection(input_symptoms))
        if not matched_symptoms: return 0.0, []
        total_weight = sum(self.disease_to_symptoms[disease].values())
        matched_weight = sum(self.disease_to_symptoms[disease][s] for s in matched_symptoms)
        symptom_coverage = len(matched_symptoms) / len(input_symptoms) if input_symptoms else 0
        confidence = (matched_weight / total_weight) * 0.7 + symptom_coverage * 0.3
        return min(confidence, 1.0), matched_symptoms

    def _get_alternative_diseases(self, symptoms: List[str], excluded_disease: str, top_n: int = 3) -> List[Dict]:
        disease_scores = {}
        for symptom in symptoms:
            if symptom in self.symptom_to_disease:
                for disease, weight in self.symptom_to_disease[symptom].items():
                    if disease != excluded_disease:
                        disease_scores[disease] = disease_scores.get(disease, 0.0) + weight
        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
        alternatives = []
        for disease, score in sorted_diseases[:top_n]:
            confidence, matched = self._calculate_kg_confidence(symptoms, disease)
            alternatives.append({'disease': disease, 'confidence': confidence, 'matched_symptoms': matched})
        return alternatives

    def _explain_symptom_relevance(self, symptoms: List[str], disease: str) -> List[Dict]:
        explanations = []
        for symptom in symptoms:
            if symptom in self.symptom_to_disease and disease in self.symptom_to_disease[symptom]:
                weight = self.symptom_to_disease[symptom][disease]
                relationship = 'primary' if weight > 0.7 else 'secondary'
                explanations.append({'symptom': symptom, 'relevance': relationship, 'weight': weight,
                                     'explanation': f"This is a {relationship} symptom for {disease}"})
            else:
                explanations.append({'symptom': symptom, 'relevance': 'unknown', 'weight': 0.0,
                                     'explanation': f"This symptom is not typically associated with {disease}"})
        return explanations