def merge_context(vector_context: str, graph_context: str) -> str:
    """
    Combines vector search results and knowledge graph data.

    Graph Knowledge is listed FIRST when it contains cause data,
    so the LLM sees and prioritizes curated expert data before
    the broader (and noisier) vector search results.
    """
    has_graph = bool(
        graph_context
        and graph_context.strip()
        and graph_context.strip() != "No specific engineering knowledge found in graph."
    )

    if has_graph:
        # Prioritize graph data: place it first so the LLM weighs it heavily
        return (
            "=== GRAPH KNOWLEDGE (Expert-Curated Causes & Relationships) ===\n"
            f"{graph_context.strip()}\n\n"
            "=== VECTOR KNOWLEDGE (Document Retrieval) ===\n"
            f"{vector_context.strip()}"
        )
    else:
        return (
            "=== VECTOR KNOWLEDGE (Document Retrieval) ===\n"
            f"{vector_context.strip()}\n\n"
            "=== GRAPH KNOWLEDGE ===\n"
            "No specific graph data available for this query."
        )
