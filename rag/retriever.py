from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag.ingest import EMBEDDING_MODEL, PERSIST_DIR

@lru_cache(maxsize=1)
def get_retriever():
    # Must use the same embedding model as ingest.py -- a mismatch silently
    # produces garbage similarity scores rather than an error.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    return db   # 🔥 RETURN DB, NOT retriever
