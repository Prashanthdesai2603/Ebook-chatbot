import requests
from bs4 import BeautifulSoup
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent.parent.parent
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Example URLs (you can add more)
URLS = [
    "https://en.wikipedia.org/wiki/Injection_moulding",
    "https://www.ptonline.com/articles/what-is-scientific-molding",
    "https://www.ptonline.com/articles/injection-molding-process-basics"
]

def scrape_page(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove junk
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Try to find common content areas if get_text is too noisy
        content = soup.find('article') or soup.find('main') or soup.body
        if content:
            text = content.get_text(separator=" ")
        else:
            text = soup.get_text(separator=" ")
            
        return " ".join(text.split())
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""


def ingest_website():
    print("Scraping website content...")

    docs = []

    for url in URLS:
        print(f"Scraping {url}...")
        text = scrape_page(url)

        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": "ptonline.com",
                        "url": url,
                        "type": "website"
                    }
                )
            )
        else:
            print(f"Skipping {url} due to empty content or error.")

    if not docs:
        print("No content was scraped. Exiting.")
        return

    print(f"Splitting {len(docs)} documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    print(f"Creating embeddings and updating vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings
    )

    vectorstore.add_documents(chunks)

    print(f"✅ Ingested {len(chunks)} website chunks into vector DB")


if __name__ == "__main__":
    ingest_website()
