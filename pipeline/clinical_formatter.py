def format_for_pharmacist(data, patient_context):

    recommendation = data.get("recommendation", {})

    monitoring = recommendation.get(
        "monitoring",
        ["Not specified"]
    )

    alternatives = recommendation.get(
        "alternatives",
        ["None"]
    )

    clinical_effects = data.get(
        "clinical_effects",
        ["Not specified"]
    )

    evidence = data.get(
        "evidence",
        []
    )

    # ensure lists
    if isinstance(monitoring, str):
        monitoring = [monitoring]

    if isinstance(alternatives, str):
        alternatives = [alternatives]

    if isinstance(clinical_effects, str):
        clinical_effects = [clinical_effects]

    return f"""
=== DRUG INTERACTION REPORT ===

Drugs:
- Drug A: {data.get("drug_pair", ["N/A", "N/A"])[0]}
- Drug B: {data.get("drug_pair", ["N/A", "N/A"])[1]}

DDI Prediction:
- Severity: {data.get("clinical_severity", "UNKNOWN")}
- Confidence: {data.get("confidence_score", "N/A")} ({data.get("confidence_level", "N/A")})

Clinical Summary:
- {data.get("interaction_summary", "Not available")}

Mechanism:
- {data.get("mechanism_of_interaction", "Not available")}

Clinical Risks:
- {"\n- ".join(clinical_effects)}

Recommendation:
- Action: {recommendation.get("action", "Not specified")}
- Monitoring: {", ".join(monitoring)}
- Alternative: {", ".join(alternatives)}

Patient Context:
- Age: {patient_context.get("age", "N/A")}
- Conditions: {", ".join(patient_context.get("conditions", []))}
- Current Medications: {", ".join(patient_context.get("medications", []))}

Evidence:
{"".join([f"{i+1}. {e}\n" for i, e in enumerate(evidence)])}

Graph Evidence:
{data.get("graph_evidence", "N/A")}

AI Confidence:
- {data.get("confidence_score", "N/A")} ({data.get("confidence_level", "N/A")})
- Reason: {data.get("confidence_reason", "Not specified")}
"""