"""Pull FDA drug label sections into data/corpus/openfda/ for vector ingestion.

    python rag/fetch/fetch_openfda.py --insecure                  # seeded 14 + top 200 by graph degree
    python rag/fetch/fetch_openfda.py --insecure --top-n 500      # wider net
    python rag/fetch/fetch_openfda.py --insecure --drugs warfarin,aspirin

openFDA (api.fda.gov) is free, public, no API key required for this volume
(default limit: 40 req/min, 1000/day -- well above what this script uses).
It re-serves the FDA's own structured product labels, which include a
dedicated `drug_interactions` section: exactly the "what to do about it"
clinical text the graph (rag/graph.py) cannot produce, since it only
answers "what interacts, by what mechanism". Also pulls the label's
patient-population sections (pregnancy, pediatric/geriatric use, renal/
hepatic impairment), since app.py's patient_context (age, conditions) has
nothing to check itself against otherwise.

--top-n picks drugs by DEGREE IN THE INTERACTION GRAPH (how many other
drugs each one interacts with in rag/processed/interactions.json), not
randomly. High-degree drugs (antipsychotics, TCAs, anticonvulsants --
broad CYP/QT-prolongation profiles) are the ones most likely to actually
get asked about, so this concentrates a bounded fetch budget where it
matters most instead of spreading it thin over 17,430 drugs uniformly.

Each drug becomes one .md file with a provenance header, so ingest.py's
--include-drug-text-style attribution works the same way here.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.fetch._http import get

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "corpus", "openfda"
)

# Sections most likely to contain actionable DDI guidance rather than
# boilerplate. clinical_pharmacology/pharmacokinetics are skipped -- that's
# already covered by the structured enzyme/transporter overlay.
SECTIONS = [
    "drug_interactions",
    "warnings_and_cautions",
    "boxed_warning",
    "contraindications",
    "dosage_and_administration",
    "use_in_specific_populations",
    "pregnancy",
    "pediatric_use",
    "geriatric_use",
]

# The 14 seeded in data/enzyme_transporter_overlay.json, always included so
# a query against any of them exercises BOTH the structured PK reasoning
# and this narrative text end to end.
SEED_DRUGS = [
    "warfarin", "aspirin", "fluconazole", "rifampin", "simvastatin",
    "atorvastatin", "ketoconazole", "clarithromycin", "digoxin",
    "quinidine", "amiodarone", "omeprazole", "clopidogrel", "ibuprofen",
]


def top_drugs_by_graph_degree(n):
    """The n drugs with the most DrugBank-documented interactions.

    Reads rag/processed/interactions.json directly rather than building the
    full graph, since this only needs interaction counts, not the whole
    NetworkX structure -- avoids the ~9s graph-build cost for a fetch script.
    """
    from rag.graph import DRUGS_PATH, INTERACTIONS_PATH

    with open(INTERACTIONS_PATH, encoding="utf-8") as f:
        interactions = json.load(f)

    counts = Counter()
    for it in interactions:
        counts[it["drug1_id"]] += 1
        counts[it["drug2_id"]] += 1

    with open(DRUGS_PATH, encoding="utf-8") as f:
        name_by_id = {d["drug_id"]: d["name"] for d in json.load(f)}

    return [name_by_id[did] for did, _ in counts.most_common(n) if did in name_by_id]


def fetch_label(generic_name, insecure):
    url = (
        "https://api.fda.gov/drug/label.json"
        f"?search=openfda.generic_name:%22{generic_name}%22&limit=1"
    )
    resp = get(url, insecure=insecure)

    if resp.status_code != 200:
        return None

    results = resp.json().get("results")
    return results[0] if results else None


def label_to_markdown(generic_name, label):
    lines = [f"# {generic_name.title()} -- FDA label excerpts", ""]
    lines.append("_Source: openFDA (api.fda.gov), U.S. FDA structured product label._")
    lines.append("")

    found_any = False
    for section in SECTIONS:
        text = label.get(section)
        if not text:
            continue
        found_any = True
        title = section.replace("_", " ").title()
        body = " ".join(text) if isinstance(text, list) else str(text)

        # Header and body deliberately share ONE line, not a header line
        # followed by a body line. A blank-line-separated "## Header\nBody"
        # gets recursively split apart by RecursiveCharacterTextSplitter once
        # the section exceeds chunk size (bodies here run thousands of
        # chars): the header ends up isolated as its own tiny, contentless
        # chunk -- and since every drug's label uses the same section names,
        # that produces literal duplicate chunks across the whole corpus.
        lines.append(f"{title}: {body}")
        lines.append("")

    return "\n".join(lines) if found_any else None


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

    print(f"Fetching {len(drugs)} drug labels from openFDA...")

    os.makedirs(OUT_DIR, exist_ok=True)

    ok, missing = [], []

    for name in drugs:
        name = name.strip()
        try:
            label = fetch_label(name, args.insecure)
        except Exception as e:
            print(f"  {name}: ERROR ({type(e).__name__})")
            missing.append(name)
            continue

        if not label:
            print(f"  {name}: no openFDA label found")
            missing.append(name)
            continue

        md = label_to_markdown(name, label)
        if not md:
            print(f"  {name}: label found but no target sections present")
            missing.append(name)
            continue

        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name.lower())
        out_path = os.path.join(OUT_DIR, f"{safe_name}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)

        print(f"  {name}: saved ({len(md):,} chars) -> {os.path.relpath(out_path)}")
        ok.append(name)

        time.sleep(0.3)  # be polite; miles under the rate limit regardless

    print(f"\n{len(ok)}/{len(drugs)} labels saved to {os.path.relpath(OUT_DIR)}")
    if missing:
        print(f"missing: {missing}")
    print("Run `python rag/ingest.py` to embed these into the vector store.")


if __name__ == "__main__":
    main()
