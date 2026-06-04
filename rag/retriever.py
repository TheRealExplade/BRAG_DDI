from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

@st.cache_resource
def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    return db   # 🔥 RETURN DB, NOT retriever