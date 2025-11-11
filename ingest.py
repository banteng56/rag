from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
import os

def ingest_documents():
    docs_path = "docs"
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Folder '{docs_path}' tidak ditemukan. Pastikan sudah ada folder docs/ dengan file .md di dalamnya.")

    loader = DirectoryLoader(docs_path, glob="*.md")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n# ", "\n## ", "\n", ".", " "]
    )
    documents = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    persist_dir = "chroma_db"
    os.makedirs(persist_dir, exist_ok=True)
    db = Chroma.from_documents(documents, embeddings, persist_directory=persist_dir)
    db.persist()

    print(f"Ingested {len(documents)} chunks into local ChromaDB at '{persist_dir}'.")

