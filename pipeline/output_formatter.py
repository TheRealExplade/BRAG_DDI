import re

def clean_text(text):
    # remove ANSI escape sequences
    return re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', text)

def format_output(llm_response: str, context : str):
    llm_response = clean_text(llm_response)

    def extract(section):
        pattern = rf"{section}:\s*(.*)"
        match = re.search(pattern, llm_response)
        return match.group(1).strip() if match else "N/A"

    structured = {
        "explanation": extract("Explanation"),
        "mechanism": extract("Mechanism"),
        "risk": extract("Risk Level"),
        "recommendation": extract("Recommendation"),
        "evidence": extract_evidence(context),
        "reasoning": extract("Reasoning"),
        "confidence" : 75
    }
    return structured


def extract_evidence(context):
    lines = context.split("\n")
    return [
        l for l in lines
        if l and not l.startswith("#")
    ][:3]