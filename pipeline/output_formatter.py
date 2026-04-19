import re

def clean_text(text):
    # remove ANSI escape sequences
    return re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', text)

def format_output(llm_response: str, context : str):
    llm_response = clean_text(llm_response)

    def extract(section):
        pattern = rf"{section}:\s*(.*?)(?:\n[A-Z][a-zA-Z ]+:|$)"
        match = re.search(pattern, llm_response, re.DOTALL)
        return match.group(1).strip() if match else "N/A"

    structured = {
        "explanation": extract("Explanation"),
        "mechanism": extract("Mechanism"),
        "risk": extract("Risk Level"),
        "recommendation": extract("Recommendation"),
        "evidence": extract_evidence(context),
        "reasoning": extract("Reasoning"),
        "confidence": parse_confidence(extract("Confidence")),
        "confidence_reason": extract("Confidence Reason"),
        "alternatives": extract("Alternatives")
    }
    return structured


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