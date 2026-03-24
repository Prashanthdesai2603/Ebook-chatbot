def detect_question_type(question: str) -> str:
    """
    Classifies the user question based on specific keywords related to injection molding.
    
    Mapping rules (Step 4):
    If question contains: "splay", "burn mark", "warpage", "sink mark", "short shot" -> defect
    If question contains: "what is", "why is", "explain" -> concept
    If question contains: "processing temperature", "pressure", "drying" -> process
    If question contains: "list", "types", "causes" -> list
    Else: -> general
    """
    q_l = question.lower()
    
    # Defect detection
    defect_keywords = ["splay", "burn mark", "warpage", "sink mark", "short shot"]
    if any(k in q_l for k in defect_keywords):
        return "defect"
    
    # Concept detection
    concept_keywords = ["what is", "why is", "explain"]
    if any(k in q_l for k in concept_keywords):
        return "concept"
    
    # Process detection
    process_keywords = ["processing temperature", "pressure", "drying"]
    if any(k in q_l for k in process_keywords):
        return "process"
    
    # List detection
    list_keywords = ["list", "types", "causes"]
    if any(k in q_l for k in list_keywords):
        return "list"
    
    # Comparison detection
    compare_keywords = ["difference", "compare", "vs", "versus", "distinguish"]
    if any(k in q_l for k in compare_keywords):
        return "compare"
    
    # Default to general
    return "general"
