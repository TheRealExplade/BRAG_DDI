import re

import re

def clean_text(text):
    return re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', text)

def extract(label, text):

    pattern = rf"{label}\s*:\s*(.+?)(?=\n[A-Z][A-Za-z\s]+:|\Z)"

    match = re.search(pattern, text, re.DOTALL)

    return match.group(1).strip() if match else "N/A"

def format_output(llm_response, context):

    llm_response = clean_text(llm_response)

    return {
        "explanation": extract("Explanation", llm_response),
        "mechanism": extract("Mechanism", llm_response),
        "risk": extract("Risk Level", llm_response),
        "recommendation": extract("Recommendation", llm_response),
        "alternatives": extract("Alternatives", llm_response),
        "confidence": extract("Confidence", llm_response),
        "confidence_reason": extract("Confidence Reason", llm_response),
        "reasoning": extract("Reasoning", llm_response),
        "evidence": extract_evidence(context)
    }


def parse_confidence(val):
    try:
        return float(val)
    except:
        return 0.5

def extract_evidence(context):
    lines = context.split("\n")

    clean = []
    for l in lines:
        l = l.strip()

        if not l:
            continue

        if "VECTOR CONTEXT" in l or "GRAPH CONTEXT" in l:
            continue

        if l.startswith("#"):
            continue

        if len(l.split()) < 5:
            continue

        clean.append(l)

    return clean[:3]