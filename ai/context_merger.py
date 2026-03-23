def merge_context(vector_context, graph_context):
    """
    Combines vector search results and knowledge graph data.
    Prioritizes Knowledge Graph Data as per Step 5.
    """
    return f"""Primary Engineering Data (Knowledge Graph):
{graph_context}

Secondary Technical Context (Vector Retrieval):
{vector_context}
"""
