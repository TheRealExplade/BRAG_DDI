import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_db():
    texts = []
    metadatas = []

    # --- JSON ingestion ---
    with open("data/drugbank.json") as f:
        data = json.load(f)

    for entry in data:
        text = f"""
        {entry['drug1']} and {entry['drug2']} interaction:
        {entry['interaction']}.
        Mechanism: {entry['mechanism']}.
        Effects: {', '.join(entry['effects'])}.
        """

        texts.append(text)
        metadatas.append({
            "drug1": entry["drug1"],
            "drug2": entry["drug2"],
            "source": "drugbank"
        })

    # --- TXT ingestion ---
    with open("data/corpus.txt") as f:
        corpus = f.read().split("\n\n")  # split chunks

    for chunk in corpus:
        texts.append(chunk)
        metadatas.append({
            "source": "corpus"
        })

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    Chroma.from_texts(
        texts,
        embeddings,
        metadatas=metadatas,
        persist_directory="./chroma_db"
    )

if __name__ == "__main__":
    build_db()