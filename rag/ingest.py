"""Build the vector store (unstructured / narrative half of the RAG).

    python rag/ingest.py                      # curated corpus + data/corpus/*
    python rag/ingest.py --include-drug-text  # + DrugBank descriptions (~19k chunks)
    python rag/ingest.py --list               # show what would be ingested, embed nothing

Structured mechanism facts (enzymes, targets, interactions) do NOT belong
here -- they are looked up deterministically by rag/graph.py. This store is
for narrative text the graph cannot answer: clinical management, monitoring,
risk context.

Drop any extra corpus files (.txt / .md) into data/corpus/ and they are
picked up automatically.
"""

import argparse
import glob
import json
import os
import sys

# README documents `python rag/ingest.py`, which puts rag/ on sys.path rather
# than the repo root, so `from rag.graph import ...` would fail. Fix it here.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paths are resolved relative to the repo root so ingest works from any cwd.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUGBANK_PATH = os.path.join(REPO_ROOT, "data", "drugbank.json")
CORPUS_PATH = os.path.join(REPO_ROOT, "data", "corpus.txt")
CORPUS_DIR = os.path.join(REPO_ROOT, "data", "corpus")
PERSIST_DIR = os.path.join(REPO_ROOT, "chroma_db")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 truncates at 256 word-pieces. Anything longer is SILENTLY
# cut off, so text must be split before embedding, not after.
#
# Splitting is measured in TOKENS, not characters: dense pharmacology text
# ("Hirudo medicinalis", "[A246609]") tokenises far worse than prose, so a
# character budget that looks safe still overflows. 220 leaves headroom under
# the 256 limit; the overlap keeps sentences from being orphaned.
MAX_TOKENS = 220
CHUNK_OVERLAP_TOKENS = 30

# Fallback if the tokenizer cannot be loaded (e.g. offline, no cache).
FALLBACK_CHUNK_CHARS = 600
FALLBACK_OVERLAP_CHARS = 80

# Chroma rejects a single add() larger than its max batch size (5461 in
# chromadb 1.5.x). Stay well under it so ingestion scales instead of failing.
BATCH_SIZE = 2000


SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _splitter():
    """Token-aware splitter, so no chunk can overflow the embedder."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

        return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=MAX_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
            separators=SEPARATORS,
        )
    except Exception as e:
        print(f"WARNING: token-aware splitting unavailable ({e}); "
              f"falling back to {FALLBACK_CHUNK_CHARS}-char chunks")
        return RecursiveCharacterTextSplitter(
            chunk_size=FALLBACK_CHUNK_CHARS,
            chunk_overlap=FALLBACK_OVERLAP_CHARS,
            separators=SEPARATORS,
        )


def _add_split(texts, metadatas, raw, meta, splitter):
    """Split raw text and append each chunk with a copy of meta."""
    if not raw or not raw.strip():
        return

    for i, chunk in enumerate(splitter.split_text(raw)):
        stripped = chunk.strip()

        if not stripped:
            continue

        # Defense against a whole class of splitter artifact: a short
        # markdown heading (or similar structural fragment) isolated as its
        # own "chunk" with no real content. Caught concretely once already
        # -- see the header/body note in fetch_openfda.py -- this guards
        # against the same pattern from any future corpus source.
        content_only = stripped.lstrip("#").strip()
        if len(content_only) < 20:
            continue

        chunk_meta = dict(meta)
        chunk_meta["chunk"] = i
        texts.append(chunk)
        metadatas.append(chunk_meta)


def build_texts(include_drug_text=False):
    texts = []
    metadatas = []
    splitter = _splitter()

    # --- curated interaction pairs (data/drugbank.json) ---
    with open(DRUGBANK_PATH, encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        text = (
            f"{entry['drug1']} and {entry['drug2']} interaction: "
            f"{entry['interaction']}. "
            f"Mechanism: {entry['mechanism']}. "
            f"Effects: {', '.join(entry['effects'])}."
        )
        _add_split(texts, metadatas, text, {
            "drug1": entry["drug1"],
            "drug2": entry["drug2"],
            "source": "drugbank",
        }, splitter)

    # --- curated free text (data/corpus.txt) ---
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for block in f.read().split("\n\n"):
            _add_split(texts, metadatas, block, {"source": "corpus"}, splitter)

    # --- anything dropped into data/corpus/ ---
    for path in sorted(glob.glob(os.path.join(CORPUS_DIR, "**", "*.*"), recursive=True)):
        if not path.lower().endswith((".txt", ".md")):
            continue
        # the drop-in instructions are not corpus content
        if os.path.basename(path).lower() == "readme.md":
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            _add_split(texts, metadatas, f.read(), {
                "source": "corpus_dir",
                "file": os.path.basename(path),
            }, splitter)

    # --- DrugBank narrative text already sitting in rag/processed/drugs.json ---
    if include_drug_text:
        from rag.graph import load_drugs

        for drug in load_drugs():
            for field in ("description", "indication", "mechanism"):
                _add_split(texts, metadatas, drug.get(field), {
                    "source": f"drugbank_{field}",
                    "drug_id": drug["drug_id"],
                    "drug_name": drug["name"],
                }, splitter)

    return texts, metadatas


def deduplicate(texts, metadatas):
    """Collapse identical chunk text to one embedding, merging metadata.

    DrugBank genuinely reuses boilerplate across many entries -- e.g. 208
    different flu-vaccine-strain drugs share the exact same description
    text. Embedding that 208 times isn't just wasteful: it actively hurts
    retrieval, since a query that matches it returns many near-duplicate
    results and crowds out more specific matches. Collapsing to one
    embedding fixes that; a merged `drug_ids`/`files` list in metadata keeps
    the fact that multiple sources share this text, rather than silently
    picking one and discarding the rest.
    """
    seen = {}
    order = []

    for text, meta in zip(texts, metadatas):
        if text not in seen:
            seen[text] = dict(meta)
            order.append(text)
            continue

        merged = seen[text]
        for key in ("drug_id", "drug1", "drug2", "file"):
            if key in meta:
                existing = merged.get(f"{key}s", merged.get(key))
                values = set(str(existing).split(",")) if existing else set()
                values.add(str(meta[key]))
                merged[f"{key}s"] = ",".join(sorted(values))
                merged.pop(key, None)

    return order, [seen[t] for t in order]


def build_db(include_drug_text=False):
    texts, metadatas = build_texts(include_drug_text=include_drug_text)

    before = len(texts)
    texts, metadatas = deduplicate(texts, metadatas)
    if before != len(texts):
        print(f"Deduplicated {before:,} -> {len(texts):,} chunks "
              f"({before - len(texts):,} exact-text duplicates merged)")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Drop any existing collection first. add_texts APPENDS, so without this
    # every re-run would duplicate the whole corpus and skew retrieval.
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    db.delete_collection()

    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    for start in range(0, len(texts), BATCH_SIZE):
        db.add_texts(
            texts[start:start + BATCH_SIZE],
            metadatas=metadatas[start:start + BATCH_SIZE],
        )
        if len(texts) > BATCH_SIZE:
            print(f"  embedded {min(start + BATCH_SIZE, len(texts)):,}/{len(texts):,}")

    print(f"Ingested {len(texts):,} chunks into {PERSIST_DIR}")


def summarise(texts, metadatas):
    by_source = {}
    for m in metadatas:
        s = m.get("source", "unknown")
        by_source[s] = by_source.get(s, 0) + 1

    print(f"{len(texts):,} chunks total")
    for s, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {s:24s} {n:7,}")

    if texts:
        longest = max(len(t) for t in texts)
        print(f"  longest chunk: {longest} chars / target {MAX_TOKENS} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-drug-text",
        action="store_true",
        help="also ingest DrugBank description/indication/mechanism text (~19k chunks, slow on CPU)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="report what would be ingested without embedding anything"
    )
    args = parser.parse_args()

    if args.list:
        summarise(*build_texts(include_drug_text=args.include_drug_text))
    else:
        build_db(include_drug_text=args.include_drug_text)
