from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"

if not VECTORSTORE_DIR.exists():
    print(f"Error: Vectorstore directory not found at {VECTORSTORE_DIR}")
    exit()

db = Chroma(
    persist_directory=str(VECTORSTORE_DIR),
    embedding_function=emb
)

docs = db.similarity_search("scientific molding", k=5)

for d in docs:
    print("\nSOURCE:", d.metadata)
    print(d.page_content[:300])
