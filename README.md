# 💊 BRAG-DDI (Hybrid RAG Drug Interaction System)

A hybrid **Vector RAG + Graph RAG + LLM pipeline** to analyze drug-drug interactions with explainable reasoning and pharmacist-ready output.

---

# 🚀 1. Setup (Do this first)

## 🔹 Install dependencies

```bash
pip install -r requirements.txt
```

If missing:

```bash
pip install langchain langchain-chroma langchain-huggingface sentence-transformers chromadb networkx pyvis requests
```

---

## 🔹 Install & Run Ollama

Download Ollama and run:

```bash
ollama run mistral
```

👉 This must stay running in background

---

# 📂 2. Project Structure

```text
BRAG_DDI/
│
├── data/
│   ├── drugbank.json
│   ├── corpus.txt
│
├── rag/
│   ├── ingest.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── graph.py
│
├── pipeline/
│   ├── main_pipeline.py
│   ├── output_formatter.py
│   ├── clinical_formatter.py
│   ├── feedback.py
│
├── llm/
│   ├── ollama_client.py
│
├── prompt/
│   ├── prompt.py
│
└── app.py (optional UI)
```

---

# 🧠 3. Build Vector Database (VERY IMPORTANT)

Run this **whenever you update data**:

```bash
python rag/ingest.py
```

👉 What it does:

* Reads `drugbank.json` + `corpus.txt`
* Creates embeddings
* Stores in `chroma_db/`

---

# ⚙️ 4. Run the Pipeline

```bash
python -m pipeline.main_pipeline
```

---

## ✅ Expected Console Output

### 🔹 Step 1: Graph context

```text
--- GRAPH CONTEXT ---
warfarin → bleeding → aspirin
```

---

### 🔹 Step 2: Retrieved docs

```text
--- RETRIEVED DOCS ---
Warfarin inhibits clotting factors...
Aspirin inhibits platelet aggregation...
```

---

### 🔹 Step 3: Raw LLM output

```text
----------------RAW OUTPUT----------------
Explanation: ...
Mechanism: ...
Risk Level: HIGH
...
```

---

### 🔹 Step 4: Structured output

```text
----------------FINAL OUTPUT----------------
{
  "drug_pair": ["warfarin", "aspirin"],
  "clinical_severity": "HIGH",
  "confidence_score": 0.82,
  ...
}
```

---

### 🔹 Step 5: Pharmacist Report

```text
🔹 DRUG INTERACTION REPORT

Drugs:
- Drug A: warfarin
- Drug B: aspirin

Clinical Summary:
- Increased bleeding risk

Recommendation:
- Avoid combination
- Monitor INR

AI Confidence:
- 0.82
- Reason: Consistent evidence
```

---

# 🔄 5. Full Pipeline Flow

```text
Input (drug1, drug2)
        ↓
Mock DDI model
        ↓
Vector RAG (Chroma retrieval)
        ↓
Reranker (improves relevance)
        ↓
Graph RAG (relationship reasoning)
        ↓
Prompt builder
        ↓
LLM (Ollama - Mistral)
        ↓
Output formatter
        ↓
Clinical formatter
        ↓
Final report
```

---

# 🧪 6. Test with different drugs

Edit this in `main_pipeline.py`:

```python
result = run_pipeline("warfarin", "aspirin")
```

Try:

```python
run_pipeline("ketoconazole", "simvastatin")
run_pipeline("ibuprofen", "warfarin")
```

---

# ⚠️ 7. Common Issues + Fixes

---

## ❌ Ollama error / crash

```text
wsarecv: connection forcibly closed
```

### ✅ Fix:

* Reduce context size (already handled)
* Restart Ollama:

```bash
ollama stop mistral
ollama run mistral
```

---

## ❌ No output / missing response

### ✅ Fix:

Check `ollama_client.py`:

* Ensure `"response"` key exists
* Print `response.json()` for debugging

---

## ❌ "INSUFFICIENT DATA"

### ✅ Fix:

* Add more data in `corpus.txt` or `drugbank.json`
* Re-run:

```bash
python rag/ingest.py
```

---

## ❌ Module not found errors

### ✅ Fix:

Run:

```bash
pip install -U langchain langchain-chroma langchain-huggingface
```

---

# 📊 8. Graph Visualization (Optional)

```bash
python visualize_graph.py
```

👉 Opens `graph.html` in browser
👉 Shows relationships between drugs, targets, effects

---

# 🧠 9. What makes this system strong

* ✅ Hybrid RAG (vector + graph)
* ✅ Reduced hallucination
* ✅ Explainable reasoning
* ✅ Pharmacist validation loop
* ✅ Modular (DDI + LLM replaceable)

---

# 🔮 10. Future Improvements

* Replace mock DDI with real model
* Fine-tuned LLM
* Graph database (Neo4j)
* Feedback-based learning loop
* UI dashboard

---

# ⚡ TL;DR Commands

```bash
# Install
pip install -r requirements.txt

# Run Ollama
ollama run mistral

# Build DB
python rag/ingest.py

# Run pipeline
python -m pipeline.main_pipeline
```

---

# 🧠 Final Note

This system is designed to:

* **minimize hallucinations**
* **maximize explainability**
* **support clinical validation**

👉 Treat it like a decision-support tool, not a replacement for experts.

---
