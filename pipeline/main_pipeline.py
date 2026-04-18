# pipeline/main_pipeline.py

from ddi.mock_ddi import get_ddi
from rag.retriever import get_retriever
from llm.ollama_client import OllamaLLM
from prompt.prompt import build_prompt
from pipeline.output_formatter import format_output
from pipeline.clinical_formatter import format_for_pharmacist
from pipeline.output_formatter import format_output
from rag.reranker import rerank

def run_pipeline(drug1, drug2):
    ddi = get_ddi(drug1, drug2)

    retriever = get_retriever()
    query = f"{drug1} {drug2} interaction clinical risk bleeding mechanism"

    docs = retriever.invoke(query)

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
        or drug2.lower() in d.page_content.lower()
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

    context = "\n".join([doc.page_content for doc in docs])

    prompt = build_prompt(ddi, context)

    llm = OllamaLLM()
    raw_output = llm.generate(prompt)

    structured_output = format_output(raw_output, context)
    structured_output["confidence"] = min(len(docs) / 2, 1.0)
    structured_output = {
        # Core identity
        "drug_pair": [drug1, drug2],

        # Clinical fields (mapped properly)
        "clinical_severity": structured_output.get("risk", "UNKNOWN"),
        "confidence_score": structured_output.get("confidence", 0.5),

        "interaction_summary": structured_output.get("explanation", ""),
        "mechanism_of_interaction": structured_output.get("mechanism", "Not available"),

        "clinical_effects": [structured_output.get("risk", "")],

        "recommendation": {
            "action": structured_output.get("recommendation", ""),
            "monitoring": ["Monitor patient"],
            "alternatives": ["Consult specialist"]
        },

        "evidence": structured_output.get("evidence", []),

        "confidence_level": "Medium",
        "confidence_reason": "Based on available retrieved evidence"
    }
    patient_context = {
        "age": 65,
        "conditions": ["Hypertension"],
        "medications": ["Aspirin"]
    }

    return structured_output


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
    save_feedback(result, feedback)