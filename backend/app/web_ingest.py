from pathlib import Path
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


BASE_DIR = Path(__file__).resolve().parent.parent.parent
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


URLS = [
    "https://en.wikipedia.org/wiki/Injection_moulding",
    "https://www.ptonline.com/articles/what-is-scientific-molding",
    "https://www.ptonline.com/articles/injection-molding-process-basics",
    "https://www.ptonline.com/articles/how-to-build-a-successful-sustainable-preventive-maintenance-program",
    "https://www.ptonline.com/articles/bulk-solid-and-melt-density-how-to-calculate-these-values-and-why-they-matter-part-1-of-2",
    "https://www.ptonline.com/articles/bulk-solid-and-melt-density-how-to-calculate-these-values-and-why-they-matter-part-2-of-2",
    "https://www.ptonline.com/articles/process-capability-and-the-hesitation-effect",
    "https://www.ptonline.com/articles/improving-molding-process-capability-understanding-the-pvt-graph-part-1",
    "https://www.ptonline.com/articles/improving-molding-process-capability-the-role-of-the-five-essential-pillars-part-2",
    "https://www.ptonline.com/news/aim-institute-continues-to-grow-plastics-education-and-training-on-10th-anniversary",
    "https://www.ptonline.com/articles/inside-the-pellet"
]


# ---------------------------------------------------------
# SELENIUM SCRAPER
# ---------------------------------------------------------
def get_driver(headless=True):
    options = Options()
    if headless:
        options.add_argument("--headless")
    
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    return driver

def scrape_with_driver(driver, url):
    try:
        print(f"Scraping: {url}")
        driver.get(url)
        time.sleep(10) 
        
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        
        article = soup.select_one("div[itemprop='articleBody']") \
               or soup.select_one(".article-body") \
               or soup.select_one(".article-content") \
               or soup.select_one("article")
        
        if article:
            text = article.get_text(separator=" ")
        else:
            text = soup.get_text(separator=" ")
            
        text = " ".join(text.split())
        print(f"Text length: {len(text)}")
        
        if len(text) < 1000:
            if "Human Verification" in text or "puzzle" in text:
                print("[!] Blocked by Human Verification. Try setting headless=False.")
            else:
                print("[!] Too little content found.")
            return ""
            
        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def ingest_website():
    print("\nStarting ingestion...")
    
    # Change headless=False if PTonline continues to block you
    driver = get_driver(headless=True)
    docs = []
    
    try:
        print("Initializing session...")
        driver.get("https://www.ptonline.com/")
        time.sleep(5)
        
        for url in URLS:
            text = scrape_with_driver(driver, url)
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": "ptonline", "url": url, "type": "website"}
                    )
                )
                print(f"[OK] Saved {url}")
            else:
                print(f"Skipped {url}")
            time.sleep(5)
            
    finally:
        driver.quit()

    if not docs:
        print("[x] No documents scraped.")
        return

    print(f"\nDocuments collected: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings
    )

    vectorstore.add_documents(chunks)

    print("\n[OK] Data stored in vector DB")
    print("Location:", VECTORSTORE_DIR)


if __name__ == "__main__":
    ingest_website()
