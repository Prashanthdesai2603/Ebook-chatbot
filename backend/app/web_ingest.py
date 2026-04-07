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
    "https://www.ptonline.com/articles/california-molding-how-to-make-it-work",
    "https://www.ptonline.com/articles/what-the-reynolds-number-means-for-injection-mold-cooling-and-how-to-achieve-it",
    "https://www.ptonline.com/articles/small-shops-big-gains-how-to-know-if-youre-ready-for-automation",
    "https://www.ptonline.com/articles/making-every-cycle-count-best-practices-in-maintenance-for-maximum-uptime",
    "https://www.ptonline.com/articles/how-to-design-a-mold-for-injection-molding-a-comprehensive-guide",
    "https://www.ptonline.com/news/plast-2026-opens-registration-offers-preview",
    "https://www.ptonline.com/news/drug-delivery-platform-launches",
    "https://www.ptonline.com/products/high-peformance-packaging-molding-platform",
    "https://www.ptonline.com/news/mgs-acquires-european-medical-injection-molder-knudsen-plast",
    "https://www.ptonline.com/products/burger-brown-brings-portability-to-cooling-line-testing-",
    "https://www.ptonline.com/news/paul-caprio-promoted-to-president-ls-mtron",
    "https://www.ptonline.com/articles/injection-molding-at-ptxpo-2026",
    "https://www.ptonline.com/articles/how-small-and-midsized-injection-molders-can-increase-efficiency-and-protect-margins",
    "https://www.ptonline.com/articles/10-years-of-plastics-technology-top-shops",
    "https://www.ptonline.com/articles/come-for-the-exhibits-stay-for-education",
    "https://www.ptonline.com/articles/using-simulation-to-anticipate-correct-core-deflection-in-injection-molding",
    "https://www.ptonline.com/articles/cobots-bring-custom-injection-molder-greater-flexibility",
    "https://www.ptonline.com/articles/injection-molding",
    "https://www.ptonline.com/articles/3d-printing-achieves-break-even-point-versus-injection-molding-at-110000-parts",
    "https://www.ptonline.com/articles/consider-the-cushion-when-seeking-shot-to-shot-consistency",
    "https://www.ptonline.com/articles/injection-molding-how-to-get-rid-of-bubbles",
    "https://www.ptonline.com/articles/improve-quality-productivity-with-advanced-screw-design",
    "https://www.ptonline.com/articles/moving-beyond-the-relative-viscosity-curve-new-method-to-find-optimum-plastic-flow-rates",
    "https://www.ptonline.com/articles/what-is-scientific-molding",
    "https://www.ptonline.com/articles/injection-molding-process-basics",
    "https://www.ptonline.com/articles/how-to-build-a-successful-sustainable-preventive-maintenance-program",
    "https://www.ptonline.com/articles/bulk-solid-and-melt-density-how-to-calculate-these-values-and-why-they-matter-part-1-of-2",
    "https://www.ptonline.com/articles/bulk-solid-and-melt-density-how-to-calculate-these-values-and-why-they-matter-part-2-of-2",
    "https://www.ptonline.com/articles/process-capability-and-the-hesitation-effect",
    "https://www.ptonline.com/articles/improving-molding-process-capability-understanding-the-pvt-graph-part-1",
    "https://www.ptonline.com/articles/improving-molding-process-capability-the-role-of-the-five-essential-pillars-part-2",
    "https://www.ptonline.com/news/aim-institute-continues-to-grow-plastics-education-and-training-on-10th-anniversary",
    "https://www.ptonline.com/articles/inside-the-pellet",
    "https://www.ptonline.com/topics/injection",
    "https://www.ptonline.com/articles/california-molding-how-to-make-it-work",
    "https://www.ptonline.com/articles/icymi-march-2026-roundup",
    "https://www.ptonline.com/articles/too-interesting-times",
    "https://www.ptonline.com/news/new-president-to-lead-kraussmaffeis-us-operations",
    "https://www.ptonline.com/articles/solving-plastics-recycling-challenges",
    "https://www.ptonline.com/articles/april-2026-volume-resin-prices-mostly-up-uncertainty-looms",
    "https://www.ptonline.com/news/progressive-components-moves-to-100-employee-ownership",
    "https://www.ptonline.com/articles/its-the-dawn-of-a-new-era-at-mid-south-extrusion",
    "https://www.ptonline.com/articles/icymi-feb-2026-roundup",
    "https://www.ptonline.com/articles/qa-sustainable-packaging-strategies-in-2026",
    "https://www.ptonline.com/articles/fail-no-more-hot-runner-valve-gate-design-eliminates-seals-cooling",
    "https://www.ptonline.com/news/mgs-acquires-european-medical-injection-molder-knudsen-plast",
    "https://www.ptonline.com/news/new-president-to-lead-kraussmaffeis-us-operations",
    "https://www.ptonline.com/articles/solving-plastics-recycling-challenges",
    "https://www.ptonline.com/articles/april-2026-volume-resin-prices-mostly-up-uncertainty-looms",
    "https://www.ptonline.com/news/progressive-components-moves-to-100-employee-ownership",
    "https://www.ptonline.com/articles/its-the-dawn-of-a-new-era-at-mid-south-extrusion",
    "https://www.ptonline.com/articles/icymi-feb-2026-roundup",
    "https://www.ptonline.com/articles/qa-sustainable-packaging-strategies-in-2026",
    "https://www.ptonline.com/articles/fail-no-more-hot-runner-valve-gate-design-eliminates-seals-cooling"

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
    driver = get_driver(headless=False)
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
