import re

def clean_text(text):
    return re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', text)

def extract(label, text):

    pattern = rf"{label}\s*:\s*(.+?)(?=\n[A-Z][A-Za-z\s]+:|\Z)"

    match = re.search(pattern, text, re.DOTALL)

    return match.group(1).strip() if match else "N/A"

def format_output(llm_response, context):

    llm_response = clean_text(llm_response)

    confidence_label = extract("Confidence", llm_response)

    return {
        "explanation": extract("Explanation", llm_response),
        "mechanism": extract("Mechanism", llm_response),
        "risk": extract("Risk Level", llm_response),
        "clinical_effects": extract_list("Clinical Effects", llm_response),
        "recommendation": extract("Recommendation", llm_response),
        "alternatives": extract_list("Alternatives", llm_response),
        "confidence_label": confidence_label,
        "confidence": parse_confidence(confidence_label),
        "confidence_reason": extract("Confidence Reason", llm_response),
        "reasoning": extract("Reasoning", llm_response),
        "evidence": extract_evidence(context)
    }


CONFIDENCE_LEVELS = {
    "LOW": 0.3,
    "MEDIUM": 0.6,
    "HIGH": 0.85,
}


def parse_confidence(val):
    if not val:
        return 0.5

    try:
        return float(val)
    except (TypeError, ValueError):
        pass

    return CONFIDENCE_LEVELS.get(str(val).strip().upper(), 0.5)


def extract_list(label, text):
    raw = extract(label, text)

    if raw == "N/A":
        return []

    return [item.strip() for item in raw.split(",") if item.strip()]

def extract_evidence(context):
    """Pull citable sentences out of the raw vector context.

    Only ever called with the vector-retrieved text (not the combined
    VECTOR+GRAPH prompt block) -- graph evidence has its own dedicated
    `graph_evidence`/`mechanism_report` output fields, so mixing it in here
    would just duplicate the same lines into two places.
    """
    lines = context.split("\n")

    clean = []
    for l in lines:
        l = l.strip()

        if not l:
            continue

        if l.startswith("#"):
            continue

        if len(l.split()) < 5:
            continue

        clean.append(l)

    return clean[:3]