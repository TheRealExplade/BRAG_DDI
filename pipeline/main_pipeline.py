from ddi.mock_ddi import get_ddi
from rag.retriever import get_retriever
from llm.ollama_client import OllamaLLM
from prompt.prompt import build_prompt
from pipeline.output_formatter import format_output
from pipeline.clinical_formatter import format_for_pharmacist
from rag.reranker import rerank
from rag.graph import load_graph
from rag.mechanism_report import build_mechanism_report, format_mechanism_report

G, lookup = load_graph()

# Chroma's default distance is L2 (unbounded, lower = closer), not cosine
# similarity. This threshold is CORPUS-SIZE-DEPENDENT -- L2 distances
# compress as the corpus grows denser, so re-check it after any large
# ingest (e.g. more openFDA drugs, DrugBank XML). Last calibrated against
# the 27k-chunk corpus (14 seed + 168 openFDA-fetched drugs + full DrugBank
# narrative text): known-relevant pairs scored 0.48-0.64, known-irrelevant
# pairs scored 0.77-0.84. 0.70 sits in that gap. Without this cutoff,
# irrelevant docs get handed to the LLM as if they were evidence --
# confirmed to cause real hallucination (see below).
VECTOR_RELEVANCE_THRESHOLD = 0.70

def run_pipeline(drug1, drug2):
    ddi = get_ddi(drug1, drug2)

    retriever = get_retriever()
    # Deliberately NOT biased toward any specific clinical concern (the
    # earlier version hardcoded "...bleeding mechanism", which dragged
    # every query toward anticoagulant content in embedding space --
    # confirmed to push genuinely relevant matches for non-bleeding pairs
    # like digoxin+quinidine or citalopram+clarithromycin below the
    # relevance threshold).
    query = f"{drug1} {drug2} drug interaction clinical significance"

    mechanism_report = build_mechanism_report(G, lookup, drug1, drug2)
    graph_context = format_mechanism_report(mechanism_report)

    print("\n--- GRAPH CONTEXT ---")
    print(graph_context)

    scored_docs = retriever.similarity_search_with_score(query, k=5)
    relevant = [(d, s) for d, s in scored_docs if s <= VECTOR_RELEVANCE_THRESHOLD]

    seen = set()
    unique_docs = []
    for d, _ in relevant:
        if d.page_content not in seen:
            unique_docs.append(d)
            seen.add(d.page_content)

    docs = rerank(query, unique_docs, boost_terms=[drug1, drug2]) if unique_docs else []

    print("\n--- RETRIEVED DOCS (relevance-filtered) ---")
    if not docs and scored_docs:
        print(f"  (best candidate scored {scored_docs[0][1]:.3f}, "
              f"above the {VECTOR_RELEVANCE_THRESHOLD} relevance threshold -- discarded)")
    for d in docs:
        print(d.page_content)

    context = "\n".join([doc.page_content for doc in docs])

    # Deterministic safety net: if NEITHER source has anything, don't ask the
    # LLM at all. Confirmed by testing that an unrelated pair (nothing in the
    # graph, nothing relevant in the vector store) still gets a confident,
    # fabricated answer out of a small local model when handed irrelevant
    # context -- e.g. it invented "warfarin" for a warfarin-free query,
    # because the prompt's "use only the provided evidence" instruction
    # can't rescue evidence that shouldn't have been provided in the first
    # place. Refusing deterministically is safer than hoping the model
    # self-reports "insufficient data".
    has_graph_evidence = bool(
        mechanism_report["documented_interactions"]
        or mechanism_report["mechanistic_overlaps"]
        or mechanism_report.get("reference_severity")
    )
    has_vector_evidence = bool(docs)

    if not has_graph_evidence and not has_vector_evidence:
        print("\n--- INSUFFICIENT EVIDENCE: skipping LLM call ---")
        return {
            "drug_pair": [drug1, drug2],
            "clinical_severity": "UNKNOWN",
            "confidence_score": 0.0,
            "confidence_level": "NONE",
            "confidence_reason": (
                "No DrugBank-documented interaction, no shared mechanistic "
                "entities, no DDInter reference, and no relevant vector "
                "evidence were found for this pair. The LLM was not called "
                "to avoid fabricating an answer with no grounding."
            ),
            "interaction_summary": "Insufficient data to assess this interaction.",
            "mechanism_of_interaction": "Not available",
            "clinical_effects": [],
            "recommendation": {
                "action": "Manual clinical review required -- no automated evidence available.",
                "alternatives": [],
            },
            "evidence": [],
            "graph_evidence": graph_context,
            "mechanism_report": mechanism_report,
        }

    # Caps sized for the 4096-token num_ctx set in OllamaLLM, leaving room
    # for the prompt template's own instructions/example (~600 tokens) and
    # the 300-token response budget. Re-check both if either grows.
    combined_context = f"""
    VECTOR:
    {context[:1500] if context else "(no relevant vector evidence found)"}

    GRAPH:
    {graph_context[:2000]}
    """

    prompt = build_prompt(ddi, combined_context)

    llm = OllamaLLM()
    try:
        raw_output = llm.generate(prompt)
    except Exception as e:
        print("LLM ERROR:", e)
        return {
            "error": "LLM_FAILURE",
            "reason": str(e)
    }
    print("----------------RAW OUTPUT----------------")
    print(raw_output)
    # Pass only the vector context, not combined_context -- graph evidence
    # is already exposed as its own `graph_evidence`/`mechanism_report`
    # fields, so extracting from it here just duplicated the same lines
    # into the generic "evidence" list a second time.
    raw_structured = format_output(raw_output, context)
    print("----------------RAW STRUCTURE----------------")
    print(raw_structured)
    final_output = {
        "drug_pair": [drug1, drug2],

        "clinical_severity": raw_structured.get("risk", "UNKNOWN"),

        "confidence_score": raw_structured.get("confidence", 0.5),
        "confidence_level": raw_structured.get("confidence_label", "N/A"),
        "confidence_reason": raw_structured.get("confidence_reason", ""),

        "interaction_summary": raw_structured.get("explanation", ""),
        "mechanism_of_interaction": raw_structured.get("mechanism", "Not available"),

        "clinical_effects": raw_structured.get("clinical_effects", []) or [raw_structured.get("risk", "")],
        "recommendation": {
            "action": raw_structured.get("recommendation", ""),
            "alternatives": raw_structured.get("alternatives", [])
        },

        "evidence": raw_structured.get("evidence", []),

        "graph_evidence": graph_context,
        "mechanism_report": mechanism_report
    }

    print("----------------FINAL OUTPUT----------------")
    print(final_output)

    return final_output

if __name__ == "__main__":
    result = run_pipeline("simvastatin", "ketoconazole")

    patient_context = {
        "age": 65,
        "conditions": ["Hypertension"],
        "medications": ["Aspirin"]
    }

    from pipeline.clinical_formatter import format_for_pharmacist
    report = format_for_pharmacist(result, patient_context)
    print(report)

    # 👇 STEP 4 — DEFINE FEEDBACK (manual testing)
    feedback = {
        "decision": "approve_with_changes",
        "severity_correction": "MODERATE",
        "recommendation_edit": "Use acetaminophen instead",
        "missing_risks": "GI bleeding",
        "notes": "Mechanism incomplete",
        "confidence": "Medium"
    }

    # 👇 STEP 5 — SAVE FEEDBACK
    from pipeline.feedback import save_feedback
    #save_feedback(result, feedback)