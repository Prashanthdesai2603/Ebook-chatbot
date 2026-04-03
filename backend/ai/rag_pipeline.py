import os
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Add project root to sys.path so 'backend.*' absolute imports resolve
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import required modules
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Import custom modules from ai/ (Relative imports for linter)
from .query_classifier import detect_question_type
from .knowledge_graph import query_knowledge_graph
from .graph_db import get_graph_context
from .context_merger import merge_context
from .prompts import (
    SYSTEM_PROMPT,
    get_defect_instruction,
    get_concept_instruction,
    get_comparison_instruction,
    get_list_instruction,
    get_general_instruction,
    get_technical_issue_instruction,
)

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
        Validates the generated response.
        - Defect/list questions must have at least 4 causes in detailed mode.
        - Any question type with fewer than 3 bullet points triggers a regeneration.
        """
        if mode == "short":
            return True, ""

        if q_type in ("defect", "list"):
            # Extract causes section for defect answers
            causes_match = re.search(
                r"Possible Causes[:\s\*\-]*(.*?)(?=Data to Verify|Corrective Actions|Scientific Explanation|$)",
                answer, re.S | re.I
            )
            if causes_match:
                causes_text = causes_match.group(1).strip()
                items = re.findall(r"(?:^|\n)\s*[•\-\*]|(?:\d+\.)", causes_text)
                if len(items) < 4:
                    return False, (
                        "INSUFFICIENT CAUSES DETECTED. Your response listed fewer than 4 causes. "
                        "You MUST provide at least 4 distinct causes covering: material factors, "
                        "processing factors, machine/mold factors, and environmental factors. "
                        "Also consider: moisture content, temperature, injection speed, degradation, "
                        "venting, and contamination as applicable."
                    )
            else:
                # For list-type or general defect questions without the template header
                bullet_items = re.findall(r"(?:^|\n)\s*[•\-\*]", answer)
                if len(bullet_items) < 3:
                    return False, (
                        "Your answer is too brief. Expand with at least 4 causes or distinct points, "
                        "covering material, processing, machine/mold, and environmental factors."
                    )

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
        
        # 3. Knowledge Graph Query (Neo4j and JSON)
        # --- Get graph data from Neo4j ---
        neo4j_context = get_graph_context(query)

        # --- JSON graph fallback ---
        json_graph_context = query_knowledge_graph(query)

        # Combine both
        graph_context = f"{neo4j_context}\n{json_graph_context}".strip()
        
        # 4. Merge Context (Format as requested via utility)
        merged_context = merge_context(vector_context, graph_context)
        
        # 5. Remove internal file path leaks (Step 7: Source Cleanup)
        merged_context = re.sub(r'[A-Za-z]:\\[^ \n]*', '[Path Removed]', merged_context)
        merged_context = re.sub(r'/[^ \n]+/[^ \n]+', '[Path Removed]', merged_context)

        # 6. LLM Prompt Construction
        prompt_instructions = ""

        if mode == "short":
            prompt_instructions = (
                "\nSHORT MODE: Return ONLY 2–3 sentences. "
                "Include the single most critical engineering point. No headers, no bullets."
            )
        elif q_type == "defect":
            prompt_instructions = (
                f"\nDEFECT MODE — Follow this structure EXACTLY. Provide AT LEAST 4 CAUSES "
                f"spanning material, processing, machine/mold, and environmental factors:\n"
                f"{get_defect_instruction()}"
                f"\n\nIMPORTANT: Also consider these factor categories in your causes: "
                "moisture/drying, melt temperature, injection speed, thermal degradation, "
                "venting adequacy, gate size, contamination, and ambient humidity."
            )
        elif q_type == "concept":
            prompt_instructions = (
                f"\nCONCEPT MODE — Follow this structure EXACTLY:\n"
                f"{get_concept_instruction()}"
            )
        elif q_type == "compare":
            prompt_instructions = (
                f"\nCOMPARISON MODE — Follow this structure EXACTLY:\n"
                f"{get_comparison_instruction()}"
            )
        elif q_type == "list":
            prompt_instructions = (
                f"\nLIST MODE — Follow this structure:\n"
                f"{get_list_instruction()}"
                "\nProvide minimum 4 items with brief engineering notes per item."
            )
        elif q_type == "process":
            prompt_instructions = (
                "\nPROCESS MODE: Provide typical industrial ranges with units. "
                "State consequences of operating outside the recommended range. "
                "Note material-grade dependencies."
            )
        else:
            prompt_instructions = (
                f"\nGENERAL MODE — Follow this structure:\n"
                f"{get_general_instruction()}"
            )

        # Explicit graph context prioritization directive
        graph_priority_note = ""
        if graph_context and graph_context.strip() and graph_context.strip() != "No specific engineering knowledge found in graph.":
            graph_priority_note = (
                "\n\n⚠️  GRAPH DATABASE PRIORITY: The 'Graph Knowledge' section below contains "
                "curated expert-validated cause data. You MUST incorporate these graph-supplied causes "
                "into your Possible Causes list. Do NOT ignore this data."
            )

        full_prompt = f"""{SYSTEM_PROMPT}
{graph_priority_note}

CONTEXT (use ALL relevant data from both Vector and Graph Knowledge sections below):
{merged_context}

USER QUESTION:
{query}

QUESTION TYPE: {q_type}
MODE: {mode}
{prompt_instructions}

ABSOLUTE FINAL RULES:
- Never mention local file paths.
- Always end your response with: Source: Injection Molding Knowledge Base
"""

        # 7. Model Call
        # max_attempts is 2 (Step 3: Regenerate if validation fails)
        answer: str = ""
        
        for attempt in range(2):
            max_tokens: int = 250 if mode == "short" else 2000
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
