"""Pull DDInter's severity-labeled interaction pairs into data/ddinter/.

    python rag/fetch/fetch_ddinter.py
    python rag/fetch/fetch_ddinter.py --insecure   # sandboxed env, see _http.py

DDInter (ddinter.scbdd.com) is a free, open-access academic DDI database --
no license or application required, unlike DrugBank. It fills the gap this
project actually has: `clinical_severity` in the pipeline output currently
comes ONLY from the LLM's own guess, with nothing to check it against.
DDInter's `Level` column (Major / Moderate / Minor) is exactly that ground
truth.

Output is one raw CSV per ATC category (~13 MB total, matching the site's
own file layout) plus a merged, deduplicated ddinter_severity.json keyed by
lowercased drug name pairs -- built for a follow-up step to match against
this project's DrugBank IDs and wire into rag/mechanism_report.py.
"""

import argparse
import csv
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.fetch._http import get

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "ddinter"
)

BASE_URL = "https://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_{code}.csv"

# DDInter's own ATC-based file split.
CATEGORY_CODES = {
    "A": "alimentary tract and metabolism",
    "B": "blood and blood forming organs",
    "D": "dermatologicals",
    "H": "systemic hormonal preparations",
    "L": "antineoplastic and immunomodulating agents",
    "P": "antiparasitic products",
    "R": "respiratory system",
    "V": "various",
}


def fetch_category(code, insecure):
    url = BASE_URL.format(code=code)
    resp = get(url, insecure=insecure)
    resp.raise_for_status()
    return resp.text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--insecure", action="store_true",
                         help="skip TLS certificate verification (see rag/fetch/_http.py)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    merged = {}  # (name_a_lower, name_b_lower) sorted tuple -> record
    total_rows = 0

    for code, label in CATEGORY_CODES.items():
        try:
            csv_text = fetch_category(code, args.insecure)
        except Exception as e:
            print(f"  {code} ({label}): ERROR ({type(e).__name__})")
            continue

        raw_path = os.path.join(OUT_DIR, f"ddinter_code_{code}.csv")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(csv_text)

        reader = csv.DictReader(io.StringIO(csv_text))
        n = 0
        for row in reader:
            a, b = row["Drug_A"].strip(), row["Drug_B"].strip()
            key = tuple(sorted((a.lower(), b.lower())))
            merged[key] = {
                "drug_a": a,
                "drug_b": b,
                "level": row["Level"].strip(),
                "atc_category": label,
            }
            n += 1

        total_rows += n
        print(f"  {code} ({label}): {n:,} rows -> {os.path.relpath(raw_path)}")

    merged_path = os.path.join(OUT_DIR, "ddinter_severity.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump({
            "_meta": {
                "source": "DDInter (ddinter.scbdd.com), free/open-access, no license required",
                "fields": "level is one of Major / Moderate / Minor, as assigned by DDInter curators",
                "keyed_by": "sorted (drug_a_lower, drug_b_lower) name pairs -- NOT yet matched to this "
                            "project's DrugBank ids. See rag/mechanism_report.py for how to wire this in.",
                "row_count": len(merged),
            },
            "pairs": {f"{k[0]}|||{k[1]}": v for k, v in merged.items()},
        }, f, indent=2)

    print(f"\n{total_rows:,} raw rows -> {len(merged):,} unique pairs after dedup")
    print(f"Merged severity lookup: {os.path.relpath(merged_path)}")


if __name__ == "__main__":
    main()
