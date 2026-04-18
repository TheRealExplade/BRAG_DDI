def format_for_pharmacist(data, patient_context):
    return f"""
🔹 DRUG INTERACTION REPORT

Drugs:
- Drug A: {data["drug_pair"][0]}
- Drug B: {data["drug_pair"][1]}

DDI Prediction:
- Severity: {data["clinical_severity"]}
- Confidence: {data["confidence_score"]}

Clinical Summary:
- {data["interaction_summary"]}

Mechanism:
- {data["mechanism_of_interaction"]}

Clinical Risks:
- {"\n- ".join(data["clinical_effects"])}

Recommendation:
- Action: {data["recommendation"]["action"]}
- Monitoring: {", ".join(data["recommendation"]["monitoring"])}
- Alternative: {", ".join(data["recommendation"]["alternatives"])}

Patient Context:
- Age: {patient_context["age"]}
- Conditions: {", ".join(patient_context["conditions"])}
- Current Medications: {", ".join(patient_context["medications"])}

Evidence:
{"".join([f"{i+1}. {e}\n" for i, e in enumerate(data["evidence"])] )}

AI Confidence:
- {data["confidence_level"]}
- Reason: {data["confidence_reason"]}
"""