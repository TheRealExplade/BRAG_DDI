# BRAG-DDI — Hybrid RAG Drug-Drug Interaction System

A drug-drug interaction (DDI) analysis pipeline combining **deterministic
graph reasoning**, **vector-retrieved clinical literature**, and a **local
LLM** to produce grounded, pharmacist-ready interaction reports — with
explicit safeguards against the LLM inventing evidence that wasn't
actually given to it.

Given two drug names, it answers: *does this pair interact, by what
mechanism, how severe, what should a pharmacist do about it* — and shows
its work at every step, so nothing in the final report is an unexplained
LLM guess.

---

## 1. How it works

Two independent evidence sources feed one LLM call. They answer different
questions and neither replaces the other:

| | **Graph RAG** (structured) | **Vector RAG** (unstructured) |
|---|---|---|
| Holds | enzymes, transporters, targets, documented interaction pairs, curator severity ratings | clinical narrative — labels, monitoring guidance, dose adjustments |
| Answers | *does this pair interact, and through what mechanism* | *what should a pharmacist actually do about it* |
| Source files | `rag/processed/` (DrugBank export) + `data/enzyme_transporter_overlay.json` + `data/ddinter/` | `data/corpus/` (openFDA labels etc.) + `data/corpus.txt` + `data/drugbank.json` |
| Built by | `rag/graph.py` (in-memory NetworkX graph, built at import) | `rag/ingest.py` (Chroma vector DB, built manually, persists to `chroma_db/`) |
| Deterministic? | Yes — no LLM involved | Retrieval is deterministic; ranking uses a small cross-encoder, no LLM |

```
run_pipeline(drug1, drug2)
    │
    ├─► ddi/mock_ddi.py          placeholder DDI prediction (see §7 — replace this last)
    │
    ├─► rag/graph.py             loads the DrugBank graph (17,430 drugs, ~1.4M interaction edges)
    │     └─► rag/mechanism_report.py
    │           ├─ documented_interactions   -- pair-specific, DrugBank ground truth
    │           ├─ mechanistic_overlaps      -- shared enzymes/targets/transporters,
    │           │                               with directional PK reasoning
    │           │                               (substrate + inhibitor -> reduced clearance, etc.)
    │           └─ reference_severity        -- DDInter's independent Major/Moderate/Minor rating
    │
    ├─► rag/retriever.py + rag/reranker.py   vector search -> relevance filter -> cross-encoder rerank
    │
    ├─► prompt/prompt.py          assembles graph + vector evidence into one prompt,
    │                             explicitly labels which parts are ground truth vs. inference vs. unverified
    │
    ├─► llm/ollama_client.py      local LLM call (Ollama)
    │
    └─► pipeline/output_formatter.py + clinical_formatter.py
          -> structured JSON + pharmacist-readable report
```

**Why two independent evidence sources, not just retrieval-augmented
generation on its own:** a small local LLM handed weak or irrelevant
context will confidently make things up rather than say "I don't know" —
confirmed directly during development (see §6). The graph layer gives the
LLM verifiable facts to reason from instead of asking it to reason from
nothing, and the pipeline refuses to call the LLM at all when *neither*
source has real evidence (§6).

---

## 2. Setup

```bash
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com) and pull a model:

```bash
ollama run mistral
```
Leave this running in its own terminal — the pipeline calls `http://localhost:11434`. Any Ollama model works; `llm/ollama_client.py` defaults to `mistral`.

---

## 3. Verify everything works

```bash
python system_check.py
```

Run this any time you're unsure whether something's broken — it's the
single source of truth for "is the system healthy." 156 checks across:
dependencies, source data integrity, graph build correctness, drug-name
resolution (including aliases), vector DB integrity (no duplicates, no
truncation, correct embedding dimension), retrieval + reranking quality,
mechanism-report logic (documented / inferred / severity / unresolved
cases), PK-reasoning directionality, prompt assembly, LLM output parsing
(including malformed input), the full pipeline end-to-end, the
hallucination guard, the graph visualizer, and the feedback store.

Works with Ollama **down** — it stubs the LLM call and skips only the
model-dependent checks, so it's safe to run any time, including in CI.

```
SUMMARY
  passed:  156
  failed:  0
RESULT: ALL CHECKS PASSED
```

---

## 4. Project structure

```
BRAG_DDI/
├── ddi/
│   ├── mock_ddi.py                  placeholder DDI predictor (§7)
│   └── mock_ddi_config.py           its hardcoded values -- edit here, not the .py
├── rag/
│   ├── graph.py                     builds the DrugBank interaction graph; name resolution + aliases
│   ├── mechanism_report.py          documented interactions + mechanistic overlaps + PK inference
│   ├── severity.py                  DDInter reference-severity lookup
│   ├── retriever.py                 Chroma vector store wrapper
│   ├── reranker.py                  cross-encoder + drug-name-match reranking
│   ├── ingest.py                    builds the vector store from data/corpus* and data/drugbank.json
│   ├── processed/                   DrugBank export (drugs.json, interactions.json) -- gitignored, see §8
│   └── fetch/                       scripts that pull free external data (§8)
│       ├── fetch_openfda.py         FDA drug label sections -> data/corpus/openfda/
│       ├── fetch_ddinter.py         DDInter severity ratings -> data/ddinter/
│       └── _http.py                 shared HTTP helper
├── pipeline/
│   ├── main_pipeline.py             orchestrates the whole flow, run_pipeline(drug1, drug2)
│   ├── output_formatter.py          parses the LLM's raw text into structured fields
│   ├── clinical_formatter.py        renders the pharmacist-facing report
│   └── feedback.py                  saves pharmacist approve/reject feedback
├── llm/
│   ├── ollama_client.py             local LLM HTTP client
│   └── interface.py                 LLM interface contract
├── prompt/
│   └── prompt.py                    the actual prompt template
├── data/
│   ├── drugbank.json                small curated interaction-pair corpus (source text)
│   ├── corpus.txt                   small curated free text (source text)
│   ├── corpus/                      drop any .txt/.md here -- auto-ingested (§8)
│   ├── enzyme_transporter_overlay.json   hand-curated demo enzyme/transporter data (§9)
│   ├── ddinter/                     downloaded DDInter severity data -- gitignored, see §8
│   └── feedback_store.json          pharmacist feedback log
├── system_check.py                  the verification suite (§3)
├── app.py                           Streamlit UI
└── visualize_graph.py               renders a drug pair's graph neighborhood to HTML
```

---

## 5. Running it

**CLI, one pair, full trace:**
```bash
python -m pipeline.main_pipeline
```
Edit the drug pair at the bottom of `pipeline/main_pipeline.py`'s `if __name__ == "__main__":` block. Prints every stage: graph context, retrieved docs, raw LLM output, parsed structure, final JSON, pharmacist report.

**Python, any pair:**
```python
from pipeline.main_pipeline import run_pipeline
from pipeline.clinical_formatter import format_for_pharmacist

result = run_pipeline("simvastatin", "ketoconazole")
print(format_for_pharmacist(result, {"age": 65, "conditions": [], "medications": []}))
```

**Streamlit UI:**
```bash
streamlit run app.py
```
Type two drug names, hit Analyze, view the report, leave pharmacist feedback (saved to `data/feedback_store.json`).

**Graph visualization for a pair:**
```bash
python visualize_graph.py warfarin fluconazole
python visualize_graph.py warfarin aspirin --shared-drugs 15   # + drugs interacting with both
```
Writes `graph.html` — drugs red, targets green, enzymes orange, transporters purple, pathways blue. Deliberately does **not** draw every drug each input interacts with (a well-studied drug like warfarin interacts with ~2,000 others — including them all produces an unrenderable ~944k-edge graph).

### Example output

For `warfarin` + `fluconazole`:

```
REFERENCE SEVERITY (DDInter): Major (independent second opinion)

DOCUMENTED DRUGBANK INTERACTIONS:
- The therapeutic efficacy of Warfarin can be increased when used in combination with Fluconazole.

INFERRED MECHANISTIC OVERLAPS (not documented -- shared biology only):
- Shared enzyme CYP2C9: warfarin=['substrate'], fluconazole=['inhibitor']
    -> rule-based PK reading: Fluconazole inhibits CYP2C9; may REDUCE clearance
       of Warfarin, increasing its exposure and toxicity risk
  (+ CYP3A4, CYP2C19 -- same reasoning)
```

...feeds into a final structured report with numeric confidence, real
clinical effects (not a placeholder), and an explicit recommendation — see
`pipeline/clinical_formatter.py`'s output shape.

---

## 6. Design choices that matter (read before changing thresholds)

**The pipeline refuses to call the LLM when there's no real evidence.**
Confirmed directly during testing: handing a small local model irrelevant
retrieved context (because vector search always returns its "best
available" match, however bad) causes it to confidently invent an answer
— including, in one test, inventing a drug that wasn't even part of the
query. `pipeline/main_pipeline.py` checks `has_graph_evidence` and
`has_vector_evidence` before ever building a prompt; if both are empty, it
returns a deterministic `UNKNOWN`/confidence-`0.0` result and never
touches the LLM.

**`VECTOR_RELEVANCE_THRESHOLD` (in `main_pipeline.py`) is corpus-size
dependent.** Chroma's L2 distance compresses as the corpus grows denser —
a threshold tuned on a small corpus becomes too strict once you add a lot
more text. Last calibrated at 0.70 against a 27k-chunk corpus
(known-relevant pairs scored 0.48–0.64, known-irrelevant pairs 0.77–0.84).
**Re-run the calibration check in `system_check.py` §6 after any large
ingest** (more openFDA drugs, DrugBank narrative text, DrugBank XML).

**The reranker boosts literal drug-name matches.** The general-purpose
cross-encoder (`ms-marco-MiniLM-L-6-v2`, trained on web search relevance,
not pharmacology) was confirmed to occasionally rank a chunk about an
unrelated drug above the actually-relevant one on narrow score margins.
`rag/reranker.py` adds a small boost per literal drug-name match in the
candidate text — enough to break close ties, not enough to override a
clear semantic signal.

**The mock DDI prediction (`ddi/mock_ddi.py`) is explicitly marked
unverified in the prompt**, not presented as fact. It always returns the
same placeholder mechanism regardless of the actual drug pair (see §7) —
early on, the prompt labeled this "KNOWN DDI," and the LLM dutifully
reported the placeholder back as if it were real, including for pairs
where it was pharmacologically wrong. The prompt now calls it a
"PRELIMINARY MODEL PREDICTION (unverified)" and instructs the LLM to
ignore it unless the actual evidence corroborates it — verified to fix
this class of error.

**`Do NOT leave fields empty` was relaxed.** The original prompt forced
the LLM to always assert something for every field, which — combined with
the mock DDI issue above — pushed it toward fabricating specific claims
(e.g. the wrong CYP enzyme) rather than admitting the evidence didn't
cover something. The prompt now explicitly allows `"Not specified in the
provided evidence"` as a correct answer.

---

## 7. `ddi/mock_ddi.py` — what it is and isn't

`get_ddi()` currently returns the same hardcoded severity/mechanism/
confidence for *every* drug pair — it's a placeholder for wherever a real
DDI classifier eventually goes, kept intentionally simple. Edit
`ddi/mock_ddi_config.py` to change its values without touching pipeline
code. Because the prompt correctly labels this as unverified (§6), it
doesn't corrupt the LLM's output even in its current placeholder form —
but replacing it with a real model is still the highest-leverage
improvement available to the *prediction* side of this system (as opposed
to the *evidence* side, which the rest of this README is about).

---

## 8. Expanding the data (this is the actual bottleneck, not the code)

### Vector corpus — add text

Drop any `.txt` / `.md` file into `data/corpus/` — picked up automatically
by `python rag/ingest.py`, split by **token count** (not characters) so
nothing silently overflows the 256-token embedder, and deduplicated
against exact-text repeats (DrugBank itself reuses boilerplate across many
entries — e.g. ~200 flu-vaccine-strain drugs share one description
verbatim; deduping this measurably improves retrieval by not flooding
top-k with copies of the same generic sentence).

```bash
python rag/ingest.py --list                # dry run: see chunk counts, embed nothing
python rag/ingest.py                       # curated corpus + data/corpus/*
python rag/ingest.py --include-drug-text   # + your own DrugBank narrative text (~23k more chunks, free, no download)
```

**Already fetched, already in the corpus** (168 drugs' worth):
```bash
python rag/fetch/fetch_openfda.py --insecure                # 14 seed + top-200-by-interaction-degree
python rag/fetch/fetch_openfda.py --insecure --top-n 500     # wider net
python rag/fetch/fetch_openfda.py --insecure --drugs warfarin,aspirin
```
`--top-n` picks drugs by how many other drugs they interact with in the
DrugBank graph — not randomly, so a bounded fetch budget concentrates on
drugs actually likely to get queried (antipsychotics, TCAs,
anticonvulsants — broad interaction profiles) rather than spreading thin
over 17,430 drugs uniformly. `--insecure` skips TLS certificate
verification; only needed in sandboxed/corporate-proxy environments where
the local trust store can't reach revocation-check endpoints — try
without it first.

**Free sources still worth pulling, in priority order:**

| Source | Access | Notes |
|---|---|---|
| **Your own `drugs.json` narrative text** | Already have it | `--include-drug-text` above — zero cost, do this first |
| **openFDA, wider coverage** | Free API; **register a free key** at open.fda.gov/apis/authentication for 240 req/min & 120k/day (vs. 40/min & 1,000/day unauthenticated) | Instant signup, no approval wait — this is the cheapest way to go from 168 drugs to thousands |
| **RxClass (NLM/RxNav)** | Free, no key | ATC drug-classification codes (e.g. `warfarin -> B01AA Vitamin K antagonists`) — enables *class-level* reasoning later ("any strong CYP3A4 inhibitor") instead of needing every drug enumerated individually |
| **PubMed Central OA subset** | Free bulk/FTP | Full-text interaction studies — richer than labels, more effort to integrate |
| **LiverTox (NIH)** | Free | Hepatotoxicity-specific monographs; needs a search step first, not a direct fetch |
| DailyMed (NIH) | Free | Same underlying FDA label content as openFDA — skip, redundant |

Avoid UpToDate / Lexicomp / Micromedex — licensed, not scrapeable.
**Flockhart Table and PharmGKB/ClinPGx are both JavaScript-rendered
single-page apps with no static HTML to scrape** — confirmed inaccessible
to a plain HTTP fetch; would need real browser automation against a site
not designed for scripted access.

### Severity — already done

```bash
python rag/fetch/fetch_ddinter.py --insecure
```
Pulls DDInter's full Major/Moderate/Minor severity dataset (~160k
curated pairs, free/open-access academic database, no license needed) into
`data/ddinter/`. Already run; `data/ddinter/ddinter_severity.json` feeds
`rag/severity.py`, which is wired into every mechanism report as an
independent second opinion on severity — cross-referenced against, not
overriding, the LLM's own judgment.

---

## 9. Enzyme / transporter data — the real gap

`rag/processed/drugs.json` ships with **0% enzyme and transporter
coverage** across all 17,430 drugs, so mechanistic (shared-CYP) reasoning
can't fire from it at all. `data/enzyme_transporter_overlay.json` fills
this in for a hand-curated seed of **12 drugs** — enough to exercise and
test the reasoning path, not enough to be useful for arbitrary pairs.

```
data/enzyme_transporter_overlay.json   <- version-controlled, editable
        │
        ├─ merged onto drugs.json by rag/graph.py::load_drugs()
        │  (only ADDS fields, never overwrites -- a real DrugBank
        │   export automatically takes precedence once present)
        ├─ actions stored on GRAPH EDGES (substrate / inhibitor / inducer)
        │  -- not on the node, because the same protein (e.g. CYP2C9)
        │     can be a metabolizing enzyme for one drug and a
        │     pharmacodynamic target for another
        └─ read by rag/mechanism_report.py -> directional PK inference:
             substrate + inhibitor -> reduced clearance -> higher exposure/toxicity
             substrate + inducer   -> increased clearance -> loss of efficacy
             substrate + substrate -> possible competitive metabolism
```

**Why it isn't bigger:** the two remaining free sources with real
substrate/inhibitor/inducer data (Flockhart Table, PharmGKB) both turned
out to be unscrapeable (§8). Hand-adding more entries from memory would
mean putting unverified pharmacology claims into a clinical tool — the
same bar this project already holds itself to elsewhere (e.g. the DDInter
severity is explicitly labeled "not ground truth" in every report). This
is deliberately queued behind real data, not stalled by oversight.

### Getting the real data: DrugBank

Apply for free academic access at **go.drugbank.com → Academic Access**
(personal/institutional application, typically a few business days for
approval — this is the one source in this whole project that can't be
scripted, since it requires your own identity and license agreement).

From the full XML export, per `<drug>` element, in priority order:

| Priority | Element | Why |
|---|---|---|
| **1** | `<enzymes>` → `<name>` + `<actions>` | The actual gap (0.1% coverage today) |
| **1** | `<transporters>` → `<name>` + `<actions>` | Same gap |
| 2 | `<targets>` → `<name>` + `<actions>` | You already have target *names* (gene symbols); this adds the missing *action* (inhibitor/agonist/antagonist) |
| 2 | `<drug-interactions>` → other id + `<description>` | May be more complete than the current `interactions.json` |
| 3 | `<carriers>` → `<name>` + `<actions>` | New category — matters for protein-binding-displacement interactions |
| 3 | `<atc-codes>` / `<categories>` | Class-level generalization (see RxClass above for a free alternative right now) |

Not needed: `<products>`, `<patents>`, `<prices>`, `<manufacturers>`,
`<experimental-properties>` — none of it touches interaction reasoning.

Once you have it: replace the `drugs` block in
`data/enzyme_transporter_overlay.json` (or write a parser that regenerates
it from the XML) — **no pipeline code changes needed**, `load_drugs()`
already merges whatever's there.

---

## 10. Troubleshooting

**Ollama connection errors** — confirm `ollama run mistral` is running in
another terminal and `curl http://localhost:11434/api/tags` returns a
model list. `llm/ollama_client.py` falls back to a canned
"manual review required" response on any Ollama failure rather than
crashing the pipeline.

**`ModuleNotFoundError`** — `pip install -r requirements.txt`.

**TLS/SSL errors from the `rag/fetch/` scripts** — add `--insecure`; this
is a known issue in sandboxed/corporate-proxy dev environments where the
local machine can't reach certificate revocation-check endpoints, not
necessarily an issue with the actual data source.

**`system_check.py` fails** — read the specific `[FAIL]` line; each check
name says exactly what broke. Re-run after `python rag/ingest.py` if it's
a vector-DB integrity failure (stale DB).

**Retrieval returns nothing relevant for a pair you know should have
data** — check `VECTOR_RELEVANCE_THRESHOLD` in `main_pipeline.py` against
current corpus size (§6); it may need recalibrating after a large ingest.

---

## 11. What this system deliberately does NOT do

- Replace pharmacist judgment — every report says so explicitly
- Trust the LLM's own severity/confidence as ground truth — it's
  cross-checked against DDInter's independent rating and DrugBank's
  documented facts, and disclosed as an "independent second opinion" not
  an override
- Fabricate evidence to fill in gaps — the deterministic hallucination
  guard (§6) refuses to answer rather than guess when there's nothing to
  ground an answer in
- Pretend the placeholder DDI model or the demo enzyme overlay are real
  clinical data — both are explicitly labeled as such everywhere they
  surface, including inside the LLM prompt itself
