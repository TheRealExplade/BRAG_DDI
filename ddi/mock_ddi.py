# ddi/mock_ddi.py

def get_ddi(drug1, drug2):
    return {
        "drug1": drug1,
        "drug2": drug2,
        "severity": "HIGH",
        "mechanism": "CYP3A4 inhibition leading to increased plasma levels",
        "confidence": 0.82
    }