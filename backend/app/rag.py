import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set
import os
import re

# Add root to sys.path to allow absolute imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from backend.app.gemini_model import generate_gemini_answer
from backend.app.openai_model import generate_openai_answer
from backend.app.guardrails import guardrails

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

from backend.ai.prompts import SYSTEM_PROMPT, get_defect_instruction

# Using shared SYSTEM_PROMPT from ai/prompts.py

def detect_question_type(query: str) -> str:
    """Classifies the user question based on specific keywords related to injection molding."""
    q_l = query.lower()
    
    # Mapping rules
    if any(k in q_l for k in ["splay", "burn mark", "warpage", "sink mark", "short shot"]):
        return "defect"
    if any(k in q_l for k in ["what is", "why is", "explain"]):
        return "concept"
    if any(k in q_l for k in ["processing temperature", "pressure", "drying"]):
        return "process"
    if any(k in q_l for k in ["list", "types", "causes"]):
        return "list"
    if any(k in q_l for k in ["difference", "compare", "vs", "versus", "distinguish"]):
        return "compare"
    return "general"

class RAGPipeline:
    def __init__(self):
        print("Initializing RAG Pipeline...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        abs_vectorstore_dir = str(VECTORSTORE_DIR.resolve())
        if VECTORSTORE_DIR.exists() and os.listdir(abs_vectorstore_dir):
            self.vectorstore = Chroma(
                persist_directory=abs_vectorstore_dir,
                embedding_function=self.embeddings
            )
            print(f"Vector Store loaded from {abs_vectorstore_dir}.")
        else:
            self.vectorstore = None
            print("Vector Store NOT found. Please run ingest scripts first.")

    # ---------------------------------------------------
    def get_context(self, query: str, k: int = 4):
        """Retrieve top_k chunks from vector store."""
        if not self.vectorstore:
            return []
        try:
            return self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        except Exception as e:
            print(f"Retrieval error: {e}")
            return self.vectorstore.similarity_search_with_score(query, k=k)

    # ---------------------------------------------------
    def build_context_and_citations(self, docs_and_scores: List[Tuple]) -> Tuple[str, str]:
        """Deduplicate, truncate, and prepare citations from metadata."""
        context_parts: List[str] = []
        source_links: Set[str] = set()
        seen_chunks: Set[str] = set()

        # Remove duplicates and format context
        for doc, score in docs_and_scores:
            content = str(doc.page_content).strip()
            if content in seen_chunks:
                continue
            seen_chunks.add(content)

            context_parts.append(content)

        # Truncate context for speed/token limits
        context_text: str = "\n\n".join(context_parts)
        if len(context_text) > 8000: # Approx 2000 tokens
            limit_slice = slice(0, 8000)
            context_text = f"{context_text[limit_slice]}..."

        return context_text, ""

    # ---------------------------------------------------
    def answer_query(self, query: str, mode: str = "short") -> str:
        if not self.vectorstore:
            return "Vector database not initialized."

        # Task 3: Detect Question Type
        q_type = detect_question_type(query)

        # Task 2: Improve Vector Retrieval (top_k = 12)
        docs_and_scores = self.get_context(query, k=12)

        # 2. Strict grounding check
        if not guardrails.validate_retrieval(docs_and_scores):
            return guardrails.REFUSAL_STRING

        # 3. Build context (Task 2: Combine retrieved chunks into a single block)
        context_text, _ = self.build_context_and_citations(docs_and_scores)

        # Task 5: Remove local file paths (Sanitize context if it contains paths)
        context_text = re.sub(r'[A-Za-z]:\\[^ \n]*', '', context_text)

        # 4. Final Prompt Construction
        # Task 6: Short Mode instruction
        mode_instruction = "MODE: SHORT (Limit answer to EXACTLY 2-3 sentences)" if mode == "short" else f"MODE: DETAILED (Apply structure for {q_type.upper()})"
        
        full_prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{context_text}

QUESTION:
{query}

DETECTED QUESTION TYPE: {q_type}
USER REQUESTED MODE: {mode_instruction}

INSTRUCTIONS:
1. Use ONLY the provided context.
2. If the user is in SHORT MODE, ignore the detailed structures and provide a 2-3 sentence technical answer.
3. If in DETAILED MODE and the type is 'defect', you MUST provide AT LEAST 4 CAUSES using the structure: Problem, Possible Causes (min 4), Data to Verify, Corrective Actions, and Scientific Explanation.
4. Ensure no local file paths are mentioned.
5. End with 'Source: Injection Molding Knowledge Base'.
"""

        # 5. Model Call
        # Temperature: 0.1 for high technical accuracy/determinism
        max_tokens = 250 if mode == "short" else 1200
        
        def call_model(prompt_text):
            print(f"Calling Gemini ({mode} mode, q_type={q_type}, tokens={max_tokens})...")
            resp = generate_gemini_answer(prompt_text, temperature=0.1, max_tokens=max_tokens)
            
            # Fallback to OpenAI if Gemini fails
            if not resp or "error" in resp.lower() or "missing" in resp.lower():
                print("Gemini call failed or incomplete. Trying OpenAI fallback...")
                resp = generate_openai_answer(prompt_text, temperature=0.1, max_tokens=max_tokens)
            return resp

        answer = call_model(full_prompt)

        # Task 7: Response Validation (Defect question AND causes < 3 -> regenerate)
        if q_type == "defect" and mode != "short" and answer:
            # Heuristic check: Look for bullets or numbered items in the 'Possible Causes' section
            try:
                # Find the text between 'Possible Causes' and the next header or end
                causes_match = re.search(r"Possible Causes[:\*\*]*(.*?)(?=Data to Verify|Corrective Actions|$)", answer, re.S | re.I)
                if causes_match:
                    causes_text = causes_match.group(1).strip()
                    # Count bullets (-, *, •) or numbered items (1., 2., etc.)
                    items = re.findall(r"(?:^|\n)\s*[\-\*•\d\.]", causes_text)
                    if len(items) < 4:
                        print(f"Validation failed: Only {len(items)} causes found for defect query. Regenerating...")
                        retry_prompt = full_prompt + "\n\nCRITICAL: Your previous response had fewer than 4 causes for the defect. You MUST list at least 4 distinct scientific causes."
                        answer = call_model(retry_prompt)
            except Exception as e:
                print(f"Validation error: {e}")

        # 6. Post-processing and Guardrails
        if not answer or answer.strip() == "" or "error" in answer.lower():
            return "Failed to complex answer. Please check API connectivity."

        validated_answer = guardrails.validate_answer_overlap(answer, context_text)
        
        if validated_answer == guardrails.REFUSAL_STRING:
            # If the model explicitly said it doesn't know, respect that
            if "insufficient data" in answer.lower() or "not mention" in answer.lower():
                return answer
            return guardrails.REFUSAL_STRING

        # 7. Final Polish (Task 5: Source line and Path cleaning)
        final_answer = validated_answer.strip()
        # Double check for any missed paths
        final_answer = re.sub(r'[A-Za-z]:\\[^ \n]*', '[Path Removed]', final_answer)
        
        # Ensure correct source line (Task 5)
        if "Source: Injection Molding Knowledge Base" not in final_answer:
            final_answer += "\n\nSource: Injection Molding Knowledge Base"

        return final_answer

rag_pipeline = RAGPipeline()
