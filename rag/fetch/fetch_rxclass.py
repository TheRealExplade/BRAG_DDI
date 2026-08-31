"""Pull ATC drug-classification codes into data/rxclass_atc.json.

    python rag/fetch/fetch_rxclass.py --insecure
    python rag/fetch/fetch_rxclass.py --insecure --top-n 500
    python rag/fetch/fetch_rxclass.py --insecure --drugs warfarin,aspirin

RxNorm/RxClass (NLM's RxNav, rxnav.nlm.nih.gov) is free, public, no API key.
Two-step lookup: drug name -> RXCUI (RxNorm's own id) -> ATC classes for
that RXCUI. ATC (Anatomical Therapeutic Chemical) codes classify drugs by
what they treat and how, e.g. warfarin -> B01AA "Vitamin K antagonists".

This does NOT fill the enzyme/transporter gap (see
data/enzyme_transporter_overlay.json for that) -- it's a different kind of
structured fact: drug CLASS, not drug MECHANISM. The value is generalizing
beyond exact drug pairs. Today the graph can only reason about pairs where
both drugs are individually profiled; ATC classes are the foundation for
later reasoning like "any Vitamin K antagonist + any platelet aggregation
inhibitor" instead of needing every such pair enumerated one at a time.
That class-level reasoning is NOT implemented yet -- this script only
fetches and stores the classification data for it.

Only saves entries for drugs already resolvable in this project's own
DrugBank graph (rag.graph.resolve_drug_id), so every entry is immediately
joinable by drugbank_id with no separate name-matching step later.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.fetch._http import get
from rag.fetch.fetch_openfda import SEED_DRUGS, top_drugs_by_graph_degree

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "rxclass_atc.json"
)

RXCUI_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
RXCLASS_URL = "https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json"


def fetch_rxcui(name, insecure):
    resp = get(RXCUI_URL, insecure=insecure, params={"name": name})
    if resp.status_code != 200:
        return None
    ids = resp.json().get("idGroup", {}).get("rxnormId", [])
    return ids[0] if ids else None


def fetch_atc_classes(rxcui, insecure):
    resp = get(RXCLASS_URL, insecure=insecure, params={"rxcui": rxcui, "relaSource": "ATC"})
    if resp.status_code != 200:
        return []

    items = resp.json().get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
    seen = set()
    classes = []
    for item in items:
        c = item.get("rxclassMinConceptItem", {})
        code, name = c.get("classId"), c.get("className")
        if code and code not in seen:
            seen.add(code)
            classes.append({"code": code, "name": name})
    return classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drugs", default=None,
                         help="comma-separated generic names (overrides --top-n entirely)")
    parser.add_argument("--top-n", type=int, default=200,
                         help="also fetch the N highest-degree drugs in the interaction graph (default 200; 0 to disable)")
    parser.add_argument("--insecure", action="store_true",
                         help="skip TLS certificate verification (see rag/fetch/_http.py)")
    args = parser.parse_args()

    if args.drugs:
        drugs = [d.strip() for d in args.drugs.split(",")]
    else:
        drugs = list(SEED_DRUGS)
        if args.top_n > 0:
            for name in top_drugs_by_graph_degree(args.top_n):
                if name.lower() not in {d.lower() for d in drugs}:
                    drugs.append(name)

    print(f"Fetching ATC classes for {len(drugs)} drugs...")

    from rag.graph import load_graph, resolve_drug_id

    G, lookup = load_graph()

    entries = {}
    ok, no_rxcui, no_atc, not_in_graph = 0, 0, 0, 0

    for name in drugs:
        name = name.strip()

        drug_id = resolve_drug_id(lookup, name)
        if drug_id is None:
            not_in_graph += 1
            continue

        try:
            rxcui = fetch_rxcui(name, args.insecure)
        except Exception as e:
            print(f"  {name}: ERROR fetching RXCUI ({type(e).__name__})")
            continue

        if not rxcui:
            no_rxcui += 1
            continue

        try:
            classes = fetch_atc_classes(rxcui, args.insecure)
        except Exception as e:
            print(f"  {name}: ERROR fetching ATC classes ({type(e).__name__})")
            continue

        if not classes:
            no_atc += 1
            continue

        entries[drug_id] = {
            "name": G.nodes[drug_id].get("name", name),
            "rxcui": rxcui,
            "atc_classes": classes,
        }
        ok += 1
        print(f"  {name}: {len(classes)} ATC class(es) -> {[c['code'] for c in classes]}")

        time.sleep(0.1)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "_meta": {
                "source": "RxNorm/RxClass (rxnav.nlm.nih.gov), NLM, free/public, no key required",
                "purpose": "ATC drug classification -- foundation for future class-level reasoning, "
                           "NOT a substitute for the enzyme/transporter overlay",
                "keyed_by": "this project's own DrugBank drug_id",
                "entry_count": len(entries),
            },
            "drugs": entries,
        }, f, indent=2)

    print(f"\n{ok} drugs classified, {not_in_graph} not in our graph, "
          f"{no_rxcui} had no RxNorm match, {no_atc} had no ATC class")
    print(f"Saved to {os.path.relpath(OUT_PATH)}")


if __name__ == "__main__":
    main()
