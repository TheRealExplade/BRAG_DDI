# pipeline/main_pipeline.py

from ddi.mock_ddi import get_ddi
from rag.retriever import get_retriever
from llm.ollama_client import OllamaLLM
from prompt.prompt import build_prompt
from pipeline.output_formatter import format_output
from pipeline.clinical_formatter import format_for_pharmacist
from pipeline.output_formatter import format_output
from rag.reranker import rerank
from rag.graph import build_graph, query_graph

G = build_graph()



def run_pipeline(drug1, drug2):
    ddi = get_ddi(drug1, drug2)

    retriever = get_retriever()
    query = f"{drug1} {drug2} interaction clinical risk bleeding mechanism"

    graph_context = query_graph(G, drug1, drug2)
    #graph_context = "\n".join(graph_context.split("\n")[:2])


    print("\n--- GRAPH CONTEXT ---")
    print(graph_context)

    docs = retriever.similarity_search(query, k=5)

    seen = set()
    unique_docs = []
    for d in docs:
        if d.page_content not in seen:
            unique_docs.append(d)
            seen.add(d.page_content)
    
    docs = rerank(query, unique_docs)   # 🔥 ADD THIS
    #print("\n--- RAW RETRIEVED DOCS ---")
    #for d in docs:
    #    print(d.page_content)

    filtered_docs = [
        d for d in docs
        if drug1.lower() in d.page_content.lower()
        and drug2.lower() in d.page_content.lower()
    ]

    if not filtered_docs:
        return {
            "error": "INSUFFICIENT DATA",
            "reason": "No relevant context found for given drug pair"
        }
    docs = filtered_docs
    print("\n--- RETRIEVED DOCS ---")
    for d in docs:
        print(d.page_content)

    #MAX_DOCS = 3
    #docs = docs[:MAX_DOCS]

    context = "\n".join([doc.page_content for doc in docs])

    combined_context = f"""
    VECTOR CONTEXT:
    {context}

    GRAPH CONTEXT:
    {graph_context}
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
    raw_structured = format_output(raw_output, combined_context)
    print("----------------RAW STRUCTURE----------------")
    print(raw_structured)
    final_output = {
        "drug_pair": [drug1, drug2],

        "clinical_severity": raw_structured.get("risk", "UNKNOWN"),

        "confidence_score": raw_structured.get("confidence", 0.5),
        "confidence_reason": raw_structured.get("confidence_reason", ""),

        "interaction_summary": raw_structured.get("explanation", ""),
        "mechanism_of_interaction": raw_structured.get("mechanism", "Not available"),

        "clinical_effects": [raw_structured.get("risk", "")],
        "recommendation": {
            "action": raw_structured.get("recommendation", ""),
            "alternatives": raw_structured.get("alternatives", [])
        },

        "evidence": raw_structured.get("evidence", []),

        "graph_evidence": graph_context
    }

    print("----------------FINAL OUTPUT----------------")
    print(final_output)

    return final_output

if __name__ == "__main__":
    result = run_pipeline("warfarin", "aspirin")

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