# ddi/mock_ddi.py

from ddi.mock_ddi_config import (
    DEFAULT_SEVERITY,
    DEFAULT_MECHANISM,
    DEFAULT_CONFIDENCE,
)

def get_ddi(drug1, drug2):
    return {
        "drug1": drug1,
        "drug2": drug2,
        "severity": DEFAULT_SEVERITY,
        "mechanism": DEFAULT_MECHANISM,
        "confidence": DEFAULT_CONFIDENCE
    }