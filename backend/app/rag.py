from typing import List, Tuple
import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from .model import llm_engine
from .guardrails import guardrails

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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
            print("Vector Store loaded.")
        else:
            self.vectorstore = None
            print("Vector Store NOT found. Please run ingest.py first.")

    def get_context(self, query: str, k: int = 5) -> List[Tuple[object, float]]:
        if not self.vectorstore:
            return []
            
        # method returns List[(Document, float)]
        # using relevance_scores for normalized 0-1 range (if supported) or manually checking
        # Chroma's default similarity_search_with_relevance_scores should work well with this embedding
        try:
            docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        except Exception as e:
            print(f"Error in relevance search: {e}. Fallback to standard score.")
            docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            # Note: standard score might not match threshold logic if it's L2 distance
            # But we will stick to relevance scores as primary plan.
            
        return docs_and_scores

    def answer_query(self, query: str, mode: str = "short") -> str:
        """
        Main RAG function.
        mode: 'short' or 'detailed'
        """
        if not self.vectorstore:
            return "Error: Database not ready. Please run ingestion."

        # 1. Retrieve
        docs_and_scores = self.get_context(query)
        
        # 2. Guardrail 1: Retrieval Relevance Check
        if docs_and_scores:
            top_score = docs_and_scores[0][1]
            print(f"DEBUG: Top Retrieval Score: {top_score:.4f} (Threshold: {guardrails.SIMILARITY_THRESHOLD})")
        else:
            print("DEBUG: No docs retrieved.")
            
        if not guardrails.validate_retrieval(docs_and_scores):
            print(f"DEBUG: REJECTED by Retrieval Threshold.")
            return guardrails.REFUSAL_STRING

        # Prepare context text and sources
        context_parts = []
        sources = []
        
        for doc, score in docs_and_scores:
            context_parts.append(doc.page_content)
            # Extract metadata
            page = doc.metadata.get("page", "Unknown")
            snippet = doc.page_content[:150].replace("\n", " ") + "..."
            sources.append(f"Page {page}: {snippet}")

        context_text = "\n\n".join(context_parts)

        # 3. Construct Prompt with STRICT HARD GUARDRAILS
        # We explicitly tell it to REFUSE if context is not enough.
        
        system_rules = (
            "1. Answer ONLY from the provided CONTEXT.\n"
            "2. Do NOT use outside knowledge.\n"
            "3. If the answer is not in the CONTEXT, return EXACTLY: 'Not Mentioned in the ebook.'\n"
            "4. Do not answer questions about people, sports, politics, or general knowledge unless present in context."
        )

        system_prompt = (
            "<|system|>\n"
            "You are a helpful assistant for an ebook. Follow these strict rules:\n"
            f"{system_rules}\n"
            f"Context:\n{context_text}\n"
             "</s>\n"
        )
        
        # Mode-specific instructions
        if mode == "short":
            instruction = (
                "Provide a concise answer in 3-5 lines. "
                "Do not be vague. Be direct."
            )
            max_new_tokens = 150 
            temperature = 0.1
        else:
            instruction = (
                "Provide a comprehensive and detailed explanation. "
                "Structure your answer with:\n"
                "- A short introduction\n"
                "- Bullet points (3-6 points) for key details\n"
                "- A concluding sentence."
            )
            max_new_tokens = 600
            temperature = 0.2

        user_prompt = f"<|user|>\n{instruction}\nQuestion: {query}\n</s>\n<|assistant|>\n"
        
        full_prompt = system_prompt + user_prompt

        # 4. Generate
        answer = llm_engine.generate(full_prompt, max_new_tokens=max_new_tokens, temperature=temperature)

        # 5. Post-Processing
        
        # Cleanup "Answer:" prefix
        import re
        answer = re.sub(r'^Answer\s*:\s*', '', answer, flags=re.IGNORECASE).strip()
        
        # Remove any trailing incomplete sentences if possible (simple heuristic)
        if answer and answer[-1] not in ".!?":
             # Try to find last punctuation
             last_punct = max(answer.rfind('.'), answer.rfind('!'), answer.rfind('?'))
             if last_punct > 0:
                 answer = answer[:last_punct+1]

        # 6. Guardrail 2: Lexical Overlap Check
        # Ensure answer actually used the context words
        checked_answer = guardrails.validate_answer_overlap(answer, context_text)
        
        if checked_answer == guardrails.REFUSAL_STRING:
            print(f"DEBUG: REJECTED by Low Lexical Overlap.")
            return guardrails.REFUSAL_STRING
        else:
            print(f"DEBUG: ACCEPTED Answer. (Lexical overlap check passed)")

        # 7. Add Sources (Disabled per user request)
        # if checked_answer != guardrails.REFUSAL_STRING and sources:
        #      source_text = "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources[:3]]) # Limit to top 3
        #      checked_answer += source_text

        return checked_answer

rag_pipeline = RAGPipeline()
