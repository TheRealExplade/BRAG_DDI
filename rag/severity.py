# rag/severity.py
#
# Ground-truth interaction severity from DDInter (ddinter.scbdd.com), a
# free/open-access academic DDI database -- no license required, unlike
# DrugBank. Fetched via rag/fetch/fetch_ddinter.py into data/ddinter/.
#
# This exists because clinical_severity in the pipeline's final output used
# to come ONLY from the LLM's own guess (see pipeline/output_formatter.py's
# `risk` field), with nothing to check it against. DDInter's curator-
# assigned Major/Moderate/Minor rating is fed into the prompt as reference
# evidence, but the LLM still produces the final call -- this is a second
# opinion, not a ground-truth override, since DDInter's own severity
# criteria may not match this project's.

import json
import os

from rag.graph import reverse_aliases

DDINTER_SEVERITY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "ddinter", "ddinter_severity.json"
)

_severity_cache = None
_severity_meta = None


def load_severity_pairs():
    global _severity_cache, _severity_meta

    if _severity_cache is None:
        if not os.path.exists(DDINTER_SEVERITY_PATH):
            _severity_cache = {}
            _severity_meta = None
            return _severity_cache

        with open(DDINTER_SEVERITY_PATH, encoding="utf-8") as f:
            data = json.load(f)

        _severity_meta = data.get("_meta")
        _severity_cache = data.get("pairs", {})

    return _severity_cache


def get_severity_meta():
    if _severity_cache is None:
        load_severity_pairs()
    return _severity_meta


def _candidate_names(raw_name, canonical_name):
    """All name spellings worth trying against DDInter's name-keyed data.

    DDInter is keyed by its own drug names, which don't always match this
    project's DrugBank-derived canonical names (USAN vs INN, e.g. "Rifampin"
    vs "Rifampicin") or whatever the caller originally typed ("aspirin" vs
    "Acetylsalicylic acid"). Try all of: what was typed, the resolved
    canonical name, and any known alias pointing at that canonical name.
    """
    candidates = {raw_name.lower().strip()}

    if canonical_name:
        canonical_lower = canonical_name.lower().strip()
        candidates.add(canonical_lower)
        candidates |= reverse_aliases().get(canonical_lower, set())

    return candidates


def lookup_severity(name_a, name_b, canonical_a=None, canonical_b=None):
    """DDInter's Major/Moderate/Minor rating for a drug pair, or None.

    Tries every combination of raw/canonical/alias spelling for each drug.
    Absence means "DDInter has no curated rating for this pair" -- it does
    NOT mean the pair is safe; DDInter's coverage is not exhaustive.
    """
    pairs = load_severity_pairs()

    if not pairs:
        return None

    candidates_a = _candidate_names(name_a, canonical_a)
    candidates_b = _candidate_names(name_b, canonical_b)

    for a in candidates_a:
        for b in candidates_b:
            key = "|||".join(sorted((a, b)))
            if key in pairs:
                return pairs[key]

    return None
