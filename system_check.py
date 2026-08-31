"""
system_check.py -- end-to-end stress check for the BRAG-DDI pipeline.

Run from the repo root:

    python system_check.py

Verifies, in order:
  1. dependencies importable
  2. source data files present and well-formed
  3. graph RAG builds with sane node/edge counts and connections
  4. drug-name resolution (including aliases) works
  5. vector RAG DB is in sync with source data and has no duplicates
  6. retrieval + reranking return relevant, correctly-ordered results
  7. mechanism report handles documented / inferred / unresolved cases
  8. prompt assembly contains both evidence sections
  9. LLM output parsing survives good AND malformed input
 10. full pipeline runs (against real Ollama if up, else a stub)
 11. feedback save/load round-trips (in a temp dir, your real store is untouched)

Exit code is 0 only if every check passes.
"""

import json
import os
import shutil
import sys
import tempfile

# The HF embedding model is cached locally; don't let a flaky network hang us.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PASSED = []
FAILED = []
WARNED = []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))
    return bool(condition)


def warn(name, detail=""):
    WARNED.append(name)
    print(f"  [WARN] {name}" + (f" -- {detail}" if detail else ""))


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------
section("1. DEPENDENCIES")
# ----------------------------------------------------------------------

import importlib

for mod in ["networkx", "langchain_chroma", "langchain_huggingface",
            "sentence_transformers", "chromadb", "requests", "streamlit", "pyvis"]:
    try:
        importlib.import_module(mod)
        check(f"import {mod}", True)
    except Exception as e:
        check(f"import {mod}", False, str(e))

if FAILED:
    print("\nMissing dependencies -- run: pip install -r requirements.txt")
    sys.exit(1)


# ----------------------------------------------------------------------
section("2. SOURCE DATA")
# ----------------------------------------------------------------------

from rag.ingest import DRUGBANK_PATH, CORPUS_PATH, PERSIST_DIR, build_texts, deduplicate
from rag.graph import DRUGS_PATH, INTERACTIONS_PATH

for path in [DRUGBANK_PATH, CORPUS_PATH, DRUGS_PATH, INTERACTIONS_PATH]:
    check(f"exists {os.path.relpath(path)}", os.path.exists(path))

with open(DRUGS_PATH, encoding="utf-8") as f:
    drugs = json.load(f)
with open(INTERACTIONS_PATH, encoding="utf-8") as f:
    interactions = json.load(f)

check("drugs.json non-empty", len(drugs) > 0, f"{len(drugs)} drugs")
check("interactions.json non-empty", len(interactions) > 0, f"{len(interactions)} interactions")

required_drug_keys = {"drug_id", "name", "targets", "enzymes", "transporters", "pathways"}
missing_keys = [d.get("name") for d in drugs if not required_drug_keys.issubset(d.keys())]
check("every drug has required keys", not missing_keys, f"{len(missing_keys)} bad records")

null_ids = [d.get("name") for d in drugs if d.get("drug_id") is None]
if null_ids:
    warn("drugs with null drug_id", f"{len(null_ids)} (skipped at graph build)")

dupe_ids = len(drugs) - len({d["drug_id"] for d in drugs})
check("drug_ids unique", dupe_ids == 0, f"{dupe_ids} duplicates")


# ----------------------------------------------------------------------
section("3. GRAPH RAG BUILD")
# ----------------------------------------------------------------------

import networkx as nx
from rag.graph import load_graph, resolve_drug_id

G, lookup = load_graph()

type_counts = {}
for _, attrs in G.nodes(data=True):
    t = attrs.get("node_type", "unknown")
    type_counts[t] = type_counts.get(t, 0) + 1

rel_counts = {}
for _, _, attrs in G.edges(data=True):
    r = attrs.get("relation", "unknown")
    rel_counts[r] = rel_counts.get(r, 0) + 1

print(f"  node types: {type_counts}")
print(f"  edge relations: {rel_counts}")

check("graph has drug nodes", type_counts.get("drug", 0) > 0, f"{type_counts.get('drug', 0)}")
check("graph has target nodes", type_counts.get("target", 0) > 0, f"{type_counts.get('target', 0)}")
check("graph has pathway nodes", type_counts.get("pathway", 0) > 0, f"{type_counts.get('pathway', 0)}")
check("graph has interaction edges", rel_counts.get("interacts_with", 0) > 0,
      f"{rel_counts.get('interacts_with', 0)}")
check("graph has 'targets' edges", rel_counts.get("targets", 0) > 0, f"{rel_counts.get('targets', 0)}")

# Every interaction edge must connect nodes typed as drugs -- otherwise
# query_graph() misreads them as shared non-drug entities.
orphan_interactions = 0
for it in interactions:
    for key in ("drug1_id", "drug2_id"):
        if G.nodes.get(it[key], {}).get("node_type") != "drug":
            orphan_interactions += 1
            break
check("all interaction edges connect drug-typed nodes", orphan_interactions == 0,
      f"{orphan_interactions}/{len(interactions)} orphaned")

# Drugs present in interactions.json but missing from drugs.json are kept,
# but must be explicitly flagged as having no mechanistic profile.
incomplete = [n for n, d in G.nodes(data=True) if d.get("profile_incomplete")]
if incomplete:
    warn("drugs with no profile in drugs.json",
         f"{len(incomplete)} (e.g. {incomplete[0]}) -- interactions kept, profile flagged incomplete")

check("lookup covers all drugs", len(lookup) > 0, f"{len(lookup)} names")

# Mechanistic-overlap reasoning is only as good as the underlying coverage.
# Zero enzymes means shared-CYP reasoning can never fire, no matter the code.
from rag.graph import load_drugs, get_overlay_meta

merged = load_drugs()
print("  data coverage (drugs.json + overlay):")
for field in ["targets", "enzymes", "transporters", "pathways"]:
    have = sum(1 for d in merged if d.get(field))
    pct = 100.0 * have / len(merged)
    print(f"    {field:13s} {have:6d}/{len(merged)} ({pct:.1f}%)")
    if have == 0:
        warn(f"no {field} data available",
             f"'shared {field[:-1]}' overlaps can never be detected -- needs a richer DrugBank export")

overlay_drugs = sum(1 for d in merged if d.get("_overlay_fields"))
check("enzyme/transporter overlay applied", overlay_drugs > 0, f"{overlay_drugs} drugs enriched")
if get_overlay_meta():
    warn("enzyme/transporter data is a HAND-CURATED DEMO SEED",
         f"{overlay_drugs} drugs only -- replace with a real DrugBank export before clinical use")

# unknown node_type would silently break query_graph's filtering
unknown_types = type_counts.get("unknown", 0)
check("no untyped nodes", unknown_types == 0, f"{unknown_types} untyped")


# ----------------------------------------------------------------------
section("4. DRUG NAME RESOLUTION")
# ----------------------------------------------------------------------

resolution_cases = [
    ("warfarin", True),               # exact, lowercase
    ("Warfarin", True),               # exact, cased
    ("  WARFARIN  ", True),           # whitespace + case
    ("acetylsalicylic acid", True),   # formal name
    ("aspirin", True),                # alias -> acetylsalicylic acid
    ("Aspirin", True),                # alias, cased
    ("coumadin", True),               # brand alias
    ("totallyfakedrugxyz", False),    # must NOT resolve
]

for name, should_resolve in resolution_cases:
    got = resolve_drug_id(lookup, name)
    ok = (got is not None) == should_resolve
    check(f"resolve {name!r}", ok, f"-> {got}")

check("aspirin and acetylsalicylic acid resolve identically",
      resolve_drug_id(lookup, "aspirin") == resolve_drug_id(lookup, "acetylsalicylic acid"))


# ----------------------------------------------------------------------
section("5. VECTOR RAG DB INTEGRITY")
# ----------------------------------------------------------------------

import chromadb

# The current DB may have been built with or without --include-drug-text
# (or extra files dropped into data/corpus/), so an exact expected count
# isn't meaningful here -- check it falls in the plausible range instead.
baseline_texts, _ = deduplicate(*build_texts(include_drug_text=False))
full_texts, _ = deduplicate(*build_texts(include_drug_text=True))
print(f"  source data (deduplicated) would produce between "
      f"{len(baseline_texts):,} and {len(full_texts):,} documents "
      f"depending on --include-drug-text and data/corpus/ contents")

client = chromadb.PersistentClient(path=PERSIST_DIR)
collections = [c if isinstance(c, str) else c.name for c in client.list_collections()]
check("chroma collection exists", "langchain" in collections, f"collections={collections}")

if "langchain" in collections:
    coll = client.get_collection("langchain")
    got = coll.get(include=["documents", "metadatas"])
    docs, metas = got["documents"], got["metadatas"]

    print(f"  DB holds {len(docs)} documents ({len(set(docs))} unique)")

    check("DB has no duplicate documents", len(docs) == len(set(docs)),
          f"{len(docs) - len(set(docs))} duplicates -- re-run: python rag/ingest.py")
    check("DB count is in the plausible range for current source data",
          len(docs) >= min(len(baseline_texts), len(full_texts)) * 0.8,
          f"db={len(docs)} -- stale/empty DB? re-run: python rag/ingest.py")
    check("no empty documents", all(d.strip() for d in docs))

    sources = {}
    for m in metas:
        s = (m or {}).get("source", "none")
        sources[s] = sources.get(s, 0) + 1
    print(f"  metadata sources: {sources}")
    check("both drugbank and corpus ingested",
          sources.get("drugbank", 0) > 0 and sources.get("corpus", 0) > 0)
    if sources.get("corpus_dir", 0) == 0:
        warn("no data/corpus/ files ingested",
             "run rag/fetch/fetch_openfda.py or drop files into data/corpus/")

    # embedding dimension must match the model (all-MiniLM-L6-v2 -> 384)
    emb = coll.get(include=["embeddings"], limit=1)["embeddings"]
    if emb is not None and len(emb) > 0:
        dim = len(emb[0])
        check("embedding dim == 384 (all-MiniLM-L6-v2)", dim == 384, f"dim={dim}")


# ----------------------------------------------------------------------
section("6. RETRIEVAL + RERANKING")
# ----------------------------------------------------------------------

from rag.retriever import get_retriever
from rag.reranker import rerank

retriever = get_retriever()

query = "warfarin aspirin interaction clinical risk bleeding mechanism"
scored = retriever.similarity_search_with_score(query, k=5)

check("similarity_search returns results", len(scored) > 0, f"{len(scored)} docs")

print("  top-5 vector hits (lower distance = closer):")
for doc, score in scored:
    preview = " ".join(doc.page_content.split())[:70]
    print(f"    {score:.4f}  {preview}")

# distances must be ordered ascending -- if not, retrieval is misconfigured
dists = [s for _, s in scored]
check("results ordered by similarity", dists == sorted(dists))

top_text = scored[0][0].page_content.lower()
check("top hit is topically relevant",
      ("warfarin" in top_text and ("aspirin" in top_text or "bleeding" in top_text)),
      f"top={' '.join(scored[0][0].page_content.split())[:60]!r}")

docs = [d for d, _ in scored]
reranked = rerank(query, docs)
check("reranker returns a subset", 0 < len(reranked) <= len(docs), f"{len(docs)} -> {len(reranked)}")
check("reranked docs came from retrieval", all(d in docs for d in reranked))

print("  reranked order:")
for d in reranked:
    print(f"    {' '.join(d.page_content.split())[:70]}")

rr_text = reranked[0].page_content.lower()
check("reranked top hit still relevant", "warfarin" in rr_text or "aspirin" in rr_text)

# A clearly unrelated query should not surface the warfarin/aspirin doc first.
off_topic = retriever.similarity_search("pediatric vaccination schedule", k=1)
if off_topic and "warfarin" in off_topic[0].page_content.lower():
    warn("off-topic query still returns warfarin doc",
         "corpus is tiny, so this is expected -- not a code bug")
else:
    check("off-topic query does not return warfarin doc", True)


# ----------------------------------------------------------------------
section("7. MECHANISM REPORT (graph reasoning)")
# ----------------------------------------------------------------------

from rag.mechanism_report import build_mechanism_report, format_mechanism_report

# --- case A: a documented DrugBank interaction ---
rep = build_mechanism_report(G, lookup, "warfarin", "aspirin")
check("A/ resolves both drugs", rep["unresolved_inputs"] == [])
check("A/ finds documented interaction", len(rep["documented_interactions"]) == 1)
check("A/ builds both drug profiles", len(rep["per_drug_profile"]) == 2)
check("A/ profile has real target data",
      len(rep["per_drug_profile"]["DB00682"]["targets"]) > 0,
      f"warfarin targets={rep['per_drug_profile']['DB00682']['targets']}")
txt = format_mechanism_report(rep)
check("A/ formatted text labels evidence as DOCUMENTED", "DOCUMENTED" in txt)
print("  --- formatted ---")
for line in txt.splitlines():
    print(f"    {line}")

# --- case B: shared biology but NO documented interaction ---
found = None
drug_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "drug"]
for a in drug_nodes[:400]:
    for b in drug_nodes[:400]:
        if a >= b or G.has_edge(a, b):
            continue
        common = [n for n in nx.common_neighbors(G, a, b)
                  if G.nodes[n].get("node_type") != "drug"]
        if common:
            found = (G.nodes[a]["name"], G.nodes[b]["name"])
            break
    if found:
        break

if found:
    rep_b = build_mechanism_report(G, lookup, *found)
    check("B/ inferred-overlap pair found", True, f"{found[0]} + {found[1]}")
    check("B/ has NO documented interaction", rep_b["documented_interactions"] == [])
    check("B/ has mechanistic overlap", len(rep_b["mechanistic_overlaps"]) > 0)
    txt_b = format_mechanism_report(rep_b)
    check("B/ formatted text labels evidence as INFERRED", "INFERRED" in txt_b)
    check("B/ carries not-documented disclaimer",
          "NOT a DrugBank-documented interaction" in rep_b["mechanistic_overlaps"][0]["note"])
    print("  --- formatted ---")
    for line in txt_b.splitlines():
        print(f"    {line}")
else:
    warn("B/ no inferred-overlap pair found in sample", "skipped")

# --- case B2: directional PK inference from enzyme/transporter actions ---
pk_cases = [
    # (drug_a, drug_b, entity, kind, expected substring in the inferred effect)
    ("warfarin", "fluconazole", "CYP2C9", "shared_enzymes", "REDUCE clearance"),
    ("warfarin", "rifampin", "CYP2C9", "shared_enzymes", "INCREASE clearance"),
    ("simvastatin", "ketoconazole", "CYP3A4", "shared_enzymes", "REDUCE clearance"),
    ("digoxin", "quinidine", "ABCB1", "shared_transporters", "REDUCE clearance"),
]

for da, db_, entity, kind, expect in pk_cases:
    rep_pk = build_mechanism_report(G, lookup, da, db_)
    entries = rep_pk["mechanistic_overlaps"][0][kind] if rep_pk["mechanistic_overlaps"] else []
    match = next((e for e in entries if e["name"] == entity), None)
    check(f"PK/ {da}+{db_} finds {entity} as {kind[7:-1]}", match is not None)
    if match:
        effects = " ".join(match["likely_pk_effects"])
        check(f"PK/ {da}+{db_} infers '{expect}'", expect in effects,
              effects[:90] or "(no effect inferred)")

# CYP2C9 is also a pharmacodynamic target for other drugs -- it must still be
# classified as an ENZYME here, via the edge relation rather than node_type.
rep_cls = build_mechanism_report(G, lookup, "warfarin", "fluconazole")
enz_names = [e["name"] for e in rep_cls["mechanistic_overlaps"][0]["shared_enzymes"]]
check("PK/ CYP2C9 classified as enzyme not target", "CYP2C9" in enz_names, str(enz_names))
check("PK/ CYP2C9 absent from shared_targets",
      "CYP2C9" not in rep_cls["mechanistic_overlaps"][0]["shared_targets"])
check("PK/ provenance disclosed on overlay-derived overlap",
      "data_provenance" in rep_cls["mechanistic_overlaps"][0])

# Honesty check: warfarin+aspirin is pharmacodynamic, NOT CYP-mediated.
# The report must not invent an enzyme overlap for it.
rep_pd = build_mechanism_report(G, lookup, "warfarin", "aspirin")
pd_enz = (rep_pd["mechanistic_overlaps"][0]["shared_enzymes"]
          if rep_pd["mechanistic_overlaps"] else [])
check("PK/ no fabricated enzyme overlap for warfarin+aspirin", pd_enz == [],
      f"{[e['name'] for e in pd_enz]}")

# --- case C: unresolved drug ---
rep_c = build_mechanism_report(G, lookup, "warfarin", "totallyfakedrugxyz")
check("C/ reports unresolved input", rep_c["unresolved_inputs"] == ["totallyfakedrugxyz"])
check("C/ claims no interactions when unresolved", rep_c["documented_interactions"] == [])
check("C/ format explains the failure",
      "Could not resolve" in format_mechanism_report(rep_c))

# --- case D: same drug twice (degenerate input) ---
rep_d = build_mechanism_report(G, lookup, "warfarin", "warfarin")
check("D/ same-drug input does not crash", isinstance(rep_d, dict))
print(f"    same-drug documented_interactions={len(rep_d['documented_interactions'])}, "
      f"overlaps={len(rep_d['mechanistic_overlaps'])}")

# --- case E: DDInter reference severity, including cross-database aliasing ---
from rag.severity import load_severity_pairs, get_severity_meta

ddinter_pairs = load_severity_pairs()
if not ddinter_pairs:
    warn("DDInter severity data not present",
         "run: python rag/fetch/fetch_ddinter.py --insecure")
else:
    print(f"  DDInter pairs loaded: {len(ddinter_pairs):,} ({get_severity_meta()['source']})")

    severity_cases = [
        ("warfarin", "aspirin", "Major"),     # exact-ish, via existing alias
        ("warfarin", "rifampin", "Major"),    # DDInter uses "Rifampicin" -- INN/USAN crosswalk
        ("simvastatin", "ketoconazole", "Major"),
    ]
    for da, db_, expect in severity_cases:
        rep_e = build_mechanism_report(G, lookup, da, db_)
        got = rep_e.get("reference_severity")
        check(f"E/ {da}+{db_} reference severity = {expect}",
              got is not None and got["level"] == expect,
              str(got))

    # a pair absent from DDInter must return None, not crash or fabricate
    rep_e_missing = build_mechanism_report(G, lookup, "digoxin", "quinidine")
    check("E/ pair absent from DDInter yields None, not a guess",
          rep_e_missing.get("reference_severity") is None)

    rep_fmt = build_mechanism_report(G, lookup, "warfarin", "aspirin")
    check("E/ reference severity appears in prompt-facing text",
          "REFERENCE SEVERITY" in format_mechanism_report(rep_fmt))


# ----------------------------------------------------------------------
section("8. PROMPT ASSEMBLY")
# ----------------------------------------------------------------------

from ddi.mock_ddi import get_ddi
from prompt.prompt import build_prompt

ddi = get_ddi("warfarin", "aspirin")
check("mock DDI returns expected keys",
      {"drug1", "drug2", "severity", "mechanism", "confidence"} == set(ddi.keys()))

graph_ctx = format_mechanism_report(rep)
combined = f"\nVECTOR:\n{'x'*10}\n\nGRAPH:\n{graph_ctx}\n"
prompt = build_prompt(ddi, combined)

check("prompt includes VECTOR section", "VECTOR:" in prompt)
check("prompt includes GRAPH section", "GRAPH:" in prompt)
check("prompt includes DOCUMENTED/INFERRED guidance",
      "DOCUMENTED" in prompt and "INFERRED" in prompt)
for field in ["Explanation:", "Mechanism:", "Risk Level:", "Clinical Effects:",
              "Recommendation:", "Alternatives:", "Confidence:", "Confidence Reason:"]:
    check(f"prompt requests {field}", field in prompt)
print(f"  prompt length: {len(prompt)} chars")
if len(prompt) > 6000:
    warn("prompt is long", f"{len(prompt)} chars may strain small local models")


# ----------------------------------------------------------------------
section("9. OUTPUT PARSING (incl. malformed input)")
# ----------------------------------------------------------------------

from pipeline.output_formatter import format_output, parse_confidence

good = """
Explanation: Warfarin and aspirin increase bleeding risk.
Mechanism: Combined anticoagulant and antiplatelet effects.
Risk Level: HIGH
Clinical Effects: bleeding, elevated INR
Recommendation: Avoid concurrent use and monitor INR.
Alternatives: Acetaminophen, Celecoxib
Confidence: HIGH
Confidence Reason: Strong documented evidence.
Reasoning: Both drugs impair coagulation.
"""
p = format_output(good, "VECTOR:\nsome long line of clinical context about bleeding risk\nGRAPH:\nx")
check("parses explanation", p["explanation"].startswith("Warfarin and aspirin"))
check("parses risk level", p["risk"] == "HIGH")
check("clinical_effects is a real list", p["clinical_effects"] == ["bleeding", "elevated INR"])
check("alternatives is a real list", p["alternatives"] == ["Acetaminophen", "Celecoxib"])
check("confidence is numeric", isinstance(p["confidence"], float), f"{p['confidence']}")
check("confidence maps HIGH->0.85", p["confidence"] == 0.85)
check("confidence_label preserved", p["confidence_label"] == "HIGH")
check("evidence extracted from context", len(p["evidence"]) > 0, f"{p['evidence']}")

check("parse_confidence('LOW')", parse_confidence("LOW") == 0.3)
check("parse_confidence('medium') case-insensitive", parse_confidence("medium") == 0.6)
check("parse_confidence('0.77') numeric passthrough", parse_confidence("0.77") == 0.77)
check("parse_confidence(garbage) -> 0.5", parse_confidence("banana") == 0.5)
check("parse_confidence('') -> 0.5", parse_confidence("") == 0.5)
check("parse_confidence(None) -> 0.5", parse_confidence(None) == 0.5)

# malformed / empty LLM output must degrade gracefully, never crash
empty = format_output("", "")
check("empty LLM output does not crash", isinstance(empty, dict))
check("empty output yields N/A risk", empty["risk"] == "N/A")
check("empty output yields default confidence", empty["confidence"] == 0.5)
check("empty output yields empty effects list", empty["clinical_effects"] == [])

garbage = format_output("the model rambled without using any labels at all", "")
check("garbage LLM output does not crash", isinstance(garbage, dict))
check("garbage output yields N/A fields", garbage["explanation"] == "N/A")

# the canned fallback ollama_client returns when the LLM is unreachable
fallback = format_output("""
Explanation: Unable to generate response
Mechanism: Unknown
Risk Level: UNKNOWN
Recommendation: Manual review required
Alternatives: None
Confidence: LOW
Confidence Reason: Model generation failed
Reasoning: Ollama runtime error
""", "")
check("LLM-failure fallback parses", fallback["risk"] == "UNKNOWN")
check("LLM-failure fallback confidence is low", fallback["confidence"] == 0.3)


# ----------------------------------------------------------------------
section("10. FULL PIPELINE")
# ----------------------------------------------------------------------

import requests
import llm.ollama_client as oc
from pipeline.clinical_formatter import format_for_pharmacist

ollama_up = False
try:
    r = requests.get("http://localhost:11434/api/tags", timeout=3)
    ollama_up = r.status_code == 200
    models = [m["name"] for m in r.json().get("models", [])]
    print(f"  Ollama is UP; models available: {models}")
    default_model = oc.OllamaLLM().model
    if not any(m.split(":")[0] == default_model for m in models):
        warn(f"default model {default_model!r} not pulled",
             f"run: ollama pull {default_model}")
except Exception as e:
    print(f"  Ollama is DOWN ({type(e).__name__}) -- using a stubbed LLM response.")
    print("  (start it with `ollama run mistral` to exercise the real model)")

if not ollama_up:
    STUB = """
Explanation: Warfarin and aspirin increase bleeding risk via combined effects.
Mechanism: Warfarin inhibits clotting factors; aspirin inhibits platelets.
Risk Level: HIGH
Clinical Effects: bleeding, elevated INR
Recommendation: Avoid concurrent use unless necessary; monitor INR.
Alternatives: Acetaminophen
Confidence: HIGH
Confidence Reason: Documented DrugBank interaction plus corpus evidence.
Reasoning: Both drugs impair coagulation pathways.
"""
    oc.OllamaLLM.generate = lambda self, prompt: STUB

from pipeline.main_pipeline import run_pipeline

result = run_pipeline("warfarin", "aspirin")

check("pipeline returned a dict", isinstance(result, dict))
check("pipeline did not error", "error" not in result,
      result.get("reason", ""))

for key in ["drug_pair", "clinical_severity", "confidence_score", "confidence_level",
            "confidence_reason", "interaction_summary", "mechanism_of_interaction",
            "clinical_effects", "recommendation", "evidence", "graph_evidence",
            "mechanism_report"]:
    check(f"output has {key}", key in result)

check("confidence_score is numeric", isinstance(result["confidence_score"], (int, float)),
      f"{result['confidence_score']!r}")
check("confidence_score in [0,1]", 0.0 <= result["confidence_score"] <= 1.0)
check("clinical_effects is a list", isinstance(result["clinical_effects"], list))
check("clinical_effects is not just the risk level",
      result["clinical_effects"] != [result["clinical_severity"]],
      f"{result['clinical_effects']}")
check("recommendation.alternatives is a list",
      isinstance(result["recommendation"]["alternatives"], list))
check("graph_evidence is populated", bool(result["graph_evidence"].strip()))
check("mechanism_report is structured", isinstance(result["mechanism_report"], dict))
check("final output is JSON-serializable",
      bool(json.dumps(result)))

report = format_for_pharmacist(result, {
    "age": 65, "conditions": ["Hypertension"], "medications": ["Aspirin"]
})
check("pharmacist report renders", len(report) > 100)
check("report is ASCII-safe for Windows console",
      all(ord(c) < 128 for c in report),
      "non-ASCII chars crash print() on cp1252 terminals")
print("  --- pharmacist report ---")
for line in report.splitlines():
    print(f"    {line}")

# unknown drug must not crash the pipeline
result_bad = run_pipeline("warfarin", "totallyfakedrugxyz")
check("pipeline survives unknown drug", isinstance(result_bad, dict))
check("unknown drug surfaces resolution failure",
      "Could not resolve" in result_bad.get("graph_evidence", ""),
      result_bad.get("graph_evidence", "")[:60])

# Hallucination guard: confirmed by manual testing that handing the LLM
# weakly-relevant vector matches (Chroma's similarity_search always returns
# "top-k" regardless of quality) causes real fabrication -- a query about
# two unrelated drugs got answered with confident, specific claims about a
# THIRD drug that wasn't even asked about. VECTOR_RELEVANCE_THRESHOLD in
# main_pipeline.py filters those out, and a genuinely evidence-free pair
# short-circuits to a deterministic "insufficient evidence" result rather
# than calling the LLM at all -- this doesn't need Ollama running to verify.
import time as _time

t0 = _time.time()
result_empty = run_pipeline("alpha-methylthiofentanyl", "Coelenteramide")
dt = _time.time() - t0

check("evidence-free pair returns without calling the LLM",
      dt < 5.0, f"{dt:.1f}s (a real LLM call takes 10s+)")
check("evidence-free pair yields UNKNOWN severity, not a guess",
      result_empty["clinical_severity"] == "UNKNOWN")
check("evidence-free pair yields zero confidence, not a fabricated score",
      result_empty["confidence_score"] == 0.0)
check("evidence-free pair yields empty evidence list",
      result_empty["evidence"] == [])
check("evidence-free pair result is still JSON-serializable",
      bool(json.dumps(result_empty)))

empty_report = format_for_pharmacist(result_empty, {"age": 65, "conditions": [], "medications": []})
check("insufficient-evidence report renders without crashing", len(empty_report) > 50)
check("insufficient-evidence report is ASCII-safe", all(ord(c) < 128 for c in empty_report))


# ----------------------------------------------------------------------
section("11. GRAPH VISUALISATION")
# ----------------------------------------------------------------------

from rag.graph import get_subgraph

sg = get_subgraph(G, lookup, "warfarin", "aspirin")
check("subgraph builds", sg is not None)

# Guard the blow-up that made this unrenderable: including every interacting
# drug gave ~2.5k nodes / ~944k edges for warfarin + aspirin.
n_nodes, n_edges = sg.number_of_nodes(), sg.number_of_edges()
print(f"  warfarin+aspirin subgraph: {n_nodes} nodes / {n_edges} edges")
check("subgraph stays renderable", n_edges < 5000, f"{n_edges} edges")
check("subgraph contains both drugs",
      resolve_drug_id(lookup, "warfarin") in sg and resolve_drug_id(lookup, "aspirin") in sg)
check("subgraph excludes unrelated interacting drugs",
      sum(1 for n, d in sg.nodes(data=True) if d.get("node_type") == "drug") == 2,
      "only the two input drugs should be drug-typed by default")

sg_shared = get_subgraph(G, lookup, "warfarin", "aspirin", max_shared_drugs=10)
n_drugs_shared = sum(1 for n, d in sg_shared.nodes(data=True) if d.get("node_type") == "drug")
check("--shared-drugs adds bounded extra drugs", 2 < n_drugs_shared <= 12, f"{n_drugs_shared} drugs")

check("unresolved pair yields None", get_subgraph(G, lookup, "warfarin", "fakedrugxyz") is None)


# ----------------------------------------------------------------------
section("12. FEEDBACK ROUND-TRIP (isolated temp dir)")
# ----------------------------------------------------------------------

from pipeline.feedback import save_feedback

original_cwd = os.getcwd()
tmp = tempfile.mkdtemp(prefix="bragddi_feedback_")
try:
    os.chdir(tmp)
    os.makedirs("data", exist_ok=True)

    save_feedback(result, {"decision": "Approve", "notes": "ok", "confidence": "High"})
    save_feedback(result, {"decision": "Reject", "notes": "no", "confidence": "Low"})

    with open("data/feedback_store.json", encoding="utf-8") as f:
        stored = json.load(f)

    check("feedback appends (does not overwrite)", len(stored) == 2, f"{len(stored)} records")
    check("feedback records have ids", all("id" in r for r in stored))
    check("feedback preserves llm_output", stored[0]["llm_output"]["drug_pair"] == ["warfarin", "aspirin"])

    # corrupt the store -- must back up, not silently destroy
    with open("data/feedback_store.json", "w", encoding="utf-8") as f:
        f.write("{{{ not valid json")

    save_feedback(result, {"decision": "Approve", "notes": "after corruption", "confidence": "High"})
    check("corrupt store backed up, not lost", os.path.exists("data/feedback_store.json.corrupt"))
    with open("data/feedback_store.json", encoding="utf-8") as f:
        recovered = json.load(f)
    check("feedback store recovers after corruption", len(recovered) == 1)
finally:
    os.chdir(original_cwd)
    shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------
section("SUMMARY")
# ----------------------------------------------------------------------

print(f"  passed:  {len(PASSED)}")
print(f"  failed:  {len(FAILED)}")
print(f"  warnings:{len(WARNED)}")

if WARNED:
    print("\n  Warnings (not failures):")
    for w in WARNED:
        print(f"    - {w}")

if FAILED:
    print("\n  FAILURES:")
    for f_ in FAILED:
        print(f"    - {f_}")
    print("\nRESULT: FAILED")
    sys.exit(1)

print("\nRESULT: ALL CHECKS PASSED")
sys.exit(0)
