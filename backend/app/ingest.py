import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
# Resolves to project root from backend/app/ingest.py
BASE_DIR = Path(__file__).resolve().parent.parent.parent 
EBOOK_DIR = BASE_DIR / "data" / "ebooks"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 200

def ingest_documents():
    print(f"Checking for ebooks in {EBOOK_DIR}...")
    if not EBOOK_DIR.exists():
        EBOOK_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created {EBOOK_DIR}. Please put your PDF there.")
        return

    files = [f for f in os.listdir(EBOOK_DIR) if f.endswith(".pdf")]
    if not files:
        print("No PDF files found.")
        return

    documents = []
    for f in files:
        path = EBOOK_DIR / f
        print(f"Loading {path}...")
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        documents.extend(docs)

    print(f"Loaded {len(documents)} pages. Splitting...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks.")

    print("Creating vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Check if vectorstore exists to avoid re-creation if not needed, 
    # but for now we'll just upsert or create new.
    abs_vectorstore_dir = str(VECTORSTORE_DIR.resolve())
    
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=abs_vectorstore_dir
    )
    print(f"Vector store created at {abs_vectorstore_dir}")

if __name__ == "__main__":
    ingest_documents()
