import os
import re
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# LangChain Chroma and Embeddings
try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    print("Error: Required LangChain packages not found. Please install langchain-chroma and langchain-huggingface.")
    sys.exit(1)

# Import Gemini logic
try:
    from backend.app.gemini_model import generate_gemini_answer
except ImportError:
    print("Error: Could not find backend.app.gemini_model. Ensure sys.path is correct.")
    sys.exit(1)

# ───────────── CONFIGURATION ─────────────
ENV_PATH = ROOT_DIR / "backend" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

VECTORSTORE_DIR = ROOT_DIR / "data" / "vectorstore"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j123")

# Initialize Neo4j Driver
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# ───────────── GRAPH FUNCTIONS ─────────────

def insert_graph(material: str, issue: str, causes: list, solutions: list):
    """
    Insert or merge material, issue, and relationship data into Neo4j.
    Uses MERGE to avoid duplicates.
    """
    query = """
    MERGE (m:Material {name: $material})
    MERGE (i:Issue {name: $issue})
    MERGE (m)-[:HAS_ISSUE]->(i)

    WITH i
    FOREACH (c IN $causes |
        MERGE (cause:Cause {name: c})
        MERGE (i)-[:CAUSED_BY]->(cause)
    )

    WITH i
    FOREACH (s IN $solutions |
        MERGE (sol:Solution {name: s})
        MERGE (i)-[:HAS_SOLUTION]->(sol)
    )
    """
    try:
        with driver.session() as session:
            session.run(query, material=material, issue=issue, causes=causes, solutions=solutions)
    except Exception as e:
        print(f"Neo4j Insert Error: {e}")

def extract_graph_from_text(text: str) -> dict:
    """
    Uses LLM (Gemini) to extract structured JSON from chunk text.
    Ensures normalized issue names and short causes/solutions.
    """
    prompt = f"""
    You are an expert Injection Molding Engineering Assistant. 
    Analyze the following technical text and extract structured troubleshooting data.

    Required JSON format:
    {{
      "material": "Polymer Name (e.g. ABS, Nylon, PC) or 'General' if not specified",
      "issue": "Normalized Issue Name (e.g. Splay, Burn Marks, Warpage, Sink Mark, Short Shot)",
      "causes": ["List of 1–3 word specific causes"],
      "solutions": ["List of 1–3 word specific solutions"]
    }}

    Rules:
    - Normalize issue names (e.g., 'silver streaks' -> 'Splay', 'scorch' -> 'Burn Marks').
    - Keep causes and solutions extremely brief (1–3 words max).
    - Avoid long sentences or paragraphs.
    - If no clear material/issue relationship is found, return exactly: null
    - Return ONLY the JSON object. Do NOT include markdown code blocks (like ```json).

    Text:
    {text}
    """
    
    try:
        response = generate_gemini_answer(prompt, temperature=0.1, max_tokens=500).strip()
        
        # Clean response from any markdown blocks if the LLM ignored instructions
        if response.startswith("```"):
            # Use regex to find the first JSON object
            match = re.search(r'\{(.*)\}', response, re.DOTALL)
            if match:
                response = match.group(0)
        
        if response.lower() == "null" or not response:
            return None

        # Parse JSON safely
        data = json.loads(response)
        
        # Basic validation
        if not data.get("material") or not data.get("issue"):
            return None
            
        return data
        
    except json.JSONDecodeError:
        # One last attempt with regex if JSON parsing failed
        try:
            match = re.search(r'\{(.*)\}', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            return None
    except Exception as e:
        print(f"Extraction error: {e}")
        return None

def build_from_vector_db(batch_limit: int = 100):
    """
    Loads documents from Chroma and populates Neo4j knowledge graph.
    """
    print("--- Starting Graph Database Build ---")
    
    # 1. Load Vector Store
    if not VECTORSTORE_DIR.exists():
        print(f"Error: Vector Store not found at {VECTORSTORE_DIR}")
        return

    print("Initializing embeddings and loading vectorstore...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR.resolve()),
        embedding_function=embeddings
    )

    # 2. Extract Documents
    # get() retrieves all data stored in Chroma
    all_data = vectorstore.get()
    documents = all_data.get("documents", [])
    
    if not documents:
        print("No documents found in Vector DB.")
        return

    total_chunks = len(documents)
    process_limit = min(total_chunks, batch_limit)
    print(f"Found {total_chunks} chunks. Processing first {process_limit}...")

    # 3. Process and Insert
    entries_created = 0
    
    for i in range(process_limit):
        chunk_content = documents[i]
        
        # Step 1: Extract structure
        extracted_data = extract_graph_from_text(chunk_content)
        
        if extracted_data:
            # Step 2: Insert into Neo4j
            material = extracted_data["material"]
            issue = extracted_data["issue"]
            causes = extracted_data.get("causes", [])
            solutions = extracted_data.get("solutions", [])
            
            insert_graph(material, issue, causes, solutions)
            
            entries_created += 1
            print(f"[{i+1}/{process_limit}] Inserted: Issue={issue}, Causes={len(causes)}")
        
        # To avoid rate limits or overwhelming logs for small batches
        # time.sleep(0.1) 

    # 4. Cleanup and Report
    print("\n--- Summary ---")
    print(f"Total chunks processed: {process_limit}")
    print(f"Total graph entries created: {entries_created}")

if __name__ == "__main__":
    start_time = time.time()
    
    # Basic Neo4j connectivity check
    try:
        with driver.session() as session:
            session.run("RETURN 1")
        print("Connected to Neo4j successfully.")
    except Exception as e:
        print(f"Neo4j Connection Failed: {e}")
        sys.exit(1)

    # Run the build logic
    # Increased limit to cover all 2028 chunks
    build_from_vector_db(batch_limit=3000)
    
    driver.close()
    print(f"\nExecution finished in {time.time() - start_time:.2f} seconds.")