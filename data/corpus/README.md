# Drop corpus files here

Any `.txt` or `.md` file in this directory (including subfolders) is picked
up automatically by `python rag/ingest.py` and chunked into the vector store.

This is the **unstructured / narrative** half of the RAG. It should hold text
the graph cannot answer: clinical management, monitoring intervals, dose
adjustment, risk context, patient-population caveats.

Do NOT put structured mechanism facts here (enzymes, targets, interaction
pairs) -- those belong in `data/enzyme_transporter_overlay.json` and
`rag/processed/`, where they are looked up deterministically.

Suggested free sources: openFDA drug labels (`drug_interactions`,
`warnings`, `dosage_and_administration` sections), DailyMed SPL, PubMed
Central Open Access subset, LiverTox, MedlinePlus.

One file per drug or per topic works well. Plain prose beats tables.
