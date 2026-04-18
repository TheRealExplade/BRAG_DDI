from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
def build_db():
    with open("data/corpus.txt") as f:
        text = f.read()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_texts(docs, embeddings, persist_directory="./chroma_db")

if __name__ == "__main__":
    build_db()