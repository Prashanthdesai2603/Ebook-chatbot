from typing import List, Tuple
import re

class Guardrails:
    def __init__(self):
        # STRICT CONFIGURATION (RELAXED for Debugging)
        self.REFUSAL_STRING = "Not Mentioned in the ebook."
        self.SIMILARITY_THRESHOLD = 0.2  # Lowered from 0.35 to prevent false negatives
                                          # Note: Check console logs for actual scores seeing 0.3+ is common.
        self.MIN_WORD_OVERLAP_RATIO = 0.05 # Lowered from 0.10 to 0.05 (5%)

    def validate_retrieval(self, relevant_docs: List[Tuple[object, float]]) -> bool:
        """
        Check if the retrieved documents are relevant enough.
        Using similarity_search_with_relevance_scores:
        - Returns (doc, score) where score is typically 0 to 1 (Cosine Similarity).
        - Higher is better.
        """
        if not relevant_docs:
            return False
            
        # Check the best score (highest similarity)
        # relevant_docs is sorted by score desc usually
        best_score = relevant_docs[0][1]
        
        # Filter negative scores if any (sometimes happens with unnormalized vectors)
        if best_score < self.SIMILARITY_THRESHOLD:
            return False
            
        return True

    def validate_answer_overlap(self, answer: str, context_text: str) -> str:
        """
        Ensures the answer actually comes from the context by checking word overlap.
        If overlap is too low, it's likely a hallucination or external knowledge.
        """
        # 1. Clean and tokenize answer
        answer_words = set(re.findall(r'\w+', answer.lower()))
        if not answer_words:
            return self.REFUSAL_STRING # Empty answer?
            
        # 2. Clean and tokenize context
        context_words = set(re.findall(r'\w+', context_text.lower()))
        
        # 3. Calculate intersection
        common_words = answer_words.intersection(context_words)
        
        # 4. Remove common stop words to avoid false positives (the, a, is, etc)
        # Just a small list for basic filtering
        stop_words = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "that", "this", "it", "for", "with", "on", "as", "be", "by"}
        common_words = common_words - stop_words
        denom = len(answer_words - stop_words)
        
        if denom == 0:
            return answer # Only stop words in answer? Let it slide or check raw length.
            
        overlap_ratio = len(common_words) / denom
        
        if overlap_ratio < self.MIN_WORD_OVERLAP_RATIO:
            return self.REFUSAL_STRING
            
        return answer

guardrails = Guardrails()
