import os
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Add root to sys.path to allow imports from other directories if needed
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import required modules
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Import custom modules from ai/ (Relative imports for linter)
from .query_classifier import detect_question_type
from .knowledge_graph import query_knowledge_graph
from .context_merger import merge_context
from .prompts import SYSTEM_PROMPT, get_defect_instruction

# Import API calls from existing backend logic
try:
    from backend.app.gemini_model import generate_gemini_answer
except ImportError:
    # Fallback or dummy if not found during standalone tests
    def generate_gemini_answer(prompt, **kwargs):
        return "Gemini API Error: Module not found."

# Configuration
VECTORSTORE_DIR = ROOT_DIR / "data" / "vectorstore"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class HybridRAGPipeline:
    def __init__(self):
        print("Initializing Hybrid RAG Pipeline...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vectorstore: Chroma | None = None
        
        abs_vectorstore_dir = str(VECTORSTORE_DIR.resolve())
        if VECTORSTORE_DIR.exists() and os.listdir(abs_vectorstore_dir):
            self.vectorstore = Chroma(
                persist_directory=abs_vectorstore_dir,
                embedding_function=self.embeddings
            )
            print(f"Vector Store loaded from {abs_vectorstore_dir}.")
        else:
            self.vectorstore = None
            print("Vector Store NOT found. Running without vector context.")

    def get_vector_context(self, query: str, k: int = 12):
        """Retrieve top_k chunks from vector store (Step 2: top_k = 12)."""
        if not self.vectorstore:
            return ""
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            # Combine retrieved chunks into one context block (Step 2)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            print(f"Retrieval error: {e}")
            return ""

    def validate_response(self, query: str, answer: str, q_type: str, mode: str) -> Tuple[bool, str]:
        """
        Validates the generated response (Step 3).
        Defect questions must have at least 4 causes.
        """
        if q_type == "defect" and mode != "short":
            # Heuristic check for number of causes
            causes_match = re.search(r"Possible Causes[:\s\*\-]*(.*?)(?=Data to Verify|Corrective Actions|Scientific Explanation|$)", answer, re.S | re.I)
            if causes_match:
                causes_text = causes_match.group(1).strip()
                # Count bullets (•, -, *) or numbered items
                items = re.findall(r"(?:^|\n)\s*[•\-\*]|(?:\d+\.)", causes_text)
                if len(items) < 4:
                    return False, "Your previous response had fewer than 4 causes. Provide at least four possible causes."
        return True, ""

    def answer_query(self, query: str, mode: str = "detailed") -> str:
        """
        Main execution flow:
        User Question -> Detect Type -> Vector Search -> KG Query -> Merge -> LLM -> Validation -> Response
        """
        start_time = time.time()
        
        # 1. Detect Question Type (Step 4)
        q_type = detect_question_type(query)
        
        # 2. Vector Search (Step 2: Top 12)
        vector_context = self.get_vector_context(query, k=12)
        
        # 3. Knowledge Graph Query (Step 5 priority 1)
        graph_context = query_knowledge_graph(query)
        
        # 4. Merge Context (Step 5: Prioritize KG)
        # Context Priority: 1. Knowledge graph data, 2. Vector retrieved context
        merged_context = merge_context(vector_context, graph_context)
        
        # 5. Remove internal file path leaks (Step 7: Source Cleanup)
        merged_context = re.sub(r'[A-Za-z]:\\[^ \n]*', '[Path Removed]', merged_context)
        merged_context = re.sub(r'/[^ \n]+/[^ \n]+', '[Path Removed]', merged_context)

        # 6. LLM Prompt Construction (Step 6: Formatting)
        prompt_instructions = ""
        if q_type == "defect" and mode != "short":
            prompt_instructions = f"\nYou MUST use the following format for this defect query and provide AT LEAST 4 CAUSES:\n{get_defect_instruction()}"
        elif q_type == "concept":
            prompt_instructions = "\nProvide a clear technical explanation and why it matters."
        elif q_type == "list":
            prompt_instructions = "\nProvide concise bullet points only."
        elif q_type == "process":
            prompt_instructions = "\nProvide typical industrial ranges and mention material grade variations."
        elif q_type == "compare":
            prompt_instructions = "\nProvide the answer in a Markdown table with separate columns for comparison. Ensure clear technical comparison points."
        
        if mode == "short":
            prompt_instructions = "\nLIMIT RESPONSE TO 2-3 SENTENCES. Include most important engineering info."

        full_prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{merged_context}

USER QUESTION:
{query}

QUESTION TYPE: {q_type}
MODE: {mode}
{prompt_instructions}

Final Rule: Never mention local file paths. Always end with:
Source: Injection Molding Knowledge Base
"""

        # 7. Model Call
        # max_attempts is 2 (Step 3: Regenerate if validation fails)
        answer: str = ""
        
        for attempt in range(2):
            max_tokens: int = 250 if mode == "short" else 1500
            answer = generate_gemini_answer(full_prompt, temperature=0.1, max_tokens=max_tokens)
            
            # Step 3: Response Validation
            is_valid, validation_msg = self.validate_response(query, answer, q_type, mode)
            if is_valid:
                break
            else:
                print(f"Validation failed after attempt {attempt + 1}: {validation_msg}. Retrying...")
                full_prompt = f"{full_prompt}\n\nERROR IN PREVIOUS RESPONSE: {validation_msg}"

        # 8. Post-processing (Step 7: Source Cleanup)
        # Ensure path leak protection on output
        answer = re.sub(r'[A-Za-z]:\\[^ \n]*', '[Path Removed]', answer)
        
        # Ensure correct source line (Step 7)
        if "Source: Injection Molding Knowledge Base" not in answer:
            answer = answer.strip() + "\n\nSource: Injection Molding Knowledge Base"
        
        # Log performance (Step 8: Target < 5s)
        end_time = time.time()
        print(f"Query processed in {end_time - start_time:.2f} seconds.")
            
        return answer

# Global instance
rag_pipeline = HybridRAGPipeline()
