"""Pull comprehensive brand-name synonyms into data/rxnorm_aliases.json.

    python rag/fetch/fetch_rxnorm_aliases.py --insecure
    python rag/fetch/fetch_rxnorm_aliases.py --insecure --top-n 500

RxNorm (rxnav.nlm.nih.gov), free/public/no-key, is the actual source of
truth RxClass sits on top of -- and it solves a different problem than ATC
classification: brand-name resolution. This project's own COMMON_ALIASES
in rag/graph.py is a tiny hand-typed list (7 entries: coumadin, tylenol,
etc). RxNorm's `allrelated` endpoint returns EVERY brand name on file for
a given ingredient -- e.g. warfarin has both "Coumadin" AND "Jantoven";
only the first was in the hand-typed list.

Unlike the enzyme overlay and RxClass data, this does NOT need a separate
merge step at graph-build time with special-case logic -- the fetched
brand names are added directly as extra keys in the name->drug_id lookup
table built by rag.graph.build_name_lookup(), so "jantoven" just resolves
like any other name once this file exists.
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
    "data", "rxnorm_aliases.json"
)

RXCUI_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
ALLRELATED_URL = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allrelated.json"


def fetch_rxcui(name, insecure):
    resp = get(RXCUI_URL, insecure=insecure, params={"name": name})
    if resp.status_code != 200:
        return None
    ids = resp.json().get("idGroup", {}).get("rxnormId", [])
    return ids[0] if ids else None


def fetch_brand_names(rxcui, insecure):
    resp = get(ALLRELATED_URL.format(rxcui=rxcui), insecure=insecure)
    if resp.status_code != 200:
        return []

    groups = resp.json().get("allRelatedGroup", {}).get("conceptGroup", [])
    names = set()
    for g in groups:
        if g.get("tty") == "BN":  # Brand Name concepts specifically
            for c in g.get("conceptProperties", []):
                if c.get("name"):
                    names.add(c["name"])
    return sorted(names)


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

    print(f"Fetching brand-name synonyms for {len(drugs)} drugs...")

    from rag.graph import load_graph, resolve_drug_id

    G, lookup = load_graph()

    entries = {}
    ok, no_brands, not_in_graph = 0, 0, 0

    for name in drugs:
        name = name.strip()

        drug_id = resolve_drug_id(lookup, name)
        if drug_id is None:
            not_in_graph += 1
            continue

        try:
            rxcui = fetch_rxcui(name, args.insecure)
            brands = fetch_brand_names(rxcui, args.insecure) if rxcui else []
        except Exception as e:
            print(f"  {name}: ERROR ({type(e).__name__})")
            continue

        if not brands:
            no_brands += 1
            continue

        entries[drug_id] = {
            "canonical_name": G.nodes[drug_id].get("name", name),
            "brand_names": brands,
        }
        ok += 1
        print(f"  {name}: {brands}")

        time.sleep(0.1)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "_meta": {
                "source": "RxNorm (rxnav.nlm.nih.gov), NLM, free/public, no key required",
                "purpose": "Brand-name synonyms merged directly into the name->drug_id lookup "
                           "table (rag.graph.build_name_lookup), not a separate alias-resolution step",
                "keyed_by": "this project's own DrugBank drug_id",
                "entry_count": len(entries),
            },
            "drugs": entries,
        }, f, indent=2)

    print(f"\n{ok} drugs with brand names, {not_in_graph} not in our graph, "
          f"{no_brands} had no brand names on file")
    print(f"Saved to {os.path.relpath(OUT_PATH)}")


if __name__ == "__main__":
    main()
