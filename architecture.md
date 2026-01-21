# System Architecture: RAG + LoRA

This document explains how the two core technologies interaction requirements.

## 1. RAG (Retrieval Augmented Generation) - The "Brain"
**Role**: Source of Truth & Factual Accuracy.
**Process**:
1.  **Ingestion**: The PDF is split into chunks (~900 tokens) and embedded into the ChromaDB vector store.
2.  **Retrieval**: When a user asks a question, the system searches ChromaDB for the top 5 most similar text chunks.
3.  **Strictness**: If the similarity score is too low (distance > threshold), the system immediately returns "I don't know".
4.  **Context**: These chunks are pasted into the System Prompt.

## 2. LoRA (Low-Rank Adaptation) - The "Voice"
**Role**: Tone, Style, and Format.
**Why LoRA?**:
-   The base model (TinyLlama) is generic.
-   LoRA fine-tuning teaches it specific behavior, e.g., "Always be polite," "Use strict bullet points," or "Follow the persona of a Librarian."
-   **Critical Context**: We do NOT train LoRA on the *facts* of the book. Facts come from RAG. We train LoRA on *how to answer*.

## The Interaction Pipeline
```mermaid
graph TD
    User[User Query] --> Backend
    Backend --> RAG[RAG Retrieval (ChromaDB)]
    RAG -- "Relevant Chunks" --> PromptConstruction
    
    subgraph Guardrails
    RAG -- "Low Similarity?" --> Block[Return 'I don't know']
    end
    
    PromptConstruction -- "Query + Context + Instructions" --> LLM
    
    subgraph AI Engine
    BaseModel[TinyLlama 1.1B]
    LoRA[LoRA Adapter (Style)]
    BaseModel <--> LoRA
    end
    
    LLM --> OutputValidation
    OutputValidation -- "Hallucination Check" --> FinalResponse
```

## Guardrails
To ensure strict ebook-only compliance:
1.  **Retrieval Filter**: Drops irrelevant chunks.
2.  **Prompt Injection**: Explicit instructions to explicitly deny outside knowledge.
3.  **Post-Processing**: Checks if the answer is generated despite no context.
