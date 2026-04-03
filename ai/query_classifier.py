def detect_question_type(question: str) -> str:
    """
    Classifies the user question into one of:
    defect | concept | process | list | compare | general

    Priority order matters: more specific types are checked first.
    """
    q_l = question.lower()

    # ── Defect detection (expanded keyword set) ────────────────────────────────
    defect_keywords = [
        "splay", "silver streak", "burn mark", "burn marks", "burning",
        "warpage", "warp", "warping",
        "sink mark", "sink marks", "sinking",
        "short shot", "short fill",
        "flash", "flashing",
        "jetting", "jet mark",
        "weld line", "knit line",
        "voids", "bubble", "bubbles",
        "delamination", "delaminate",
        "discoloration", "discolouration", "black spots", "black specks",
        "surface defect", "surface finish problem",
        "gate blush", "gate mark",
        "hesitation mark",
        "flow mark", "flow lines",
    ]
    if any(k in q_l for k in defect_keywords):
        return "defect"

    # ── Comparison detection ───────────────────────────────────────────────────
    compare_keywords = ["difference", "compare", " vs ", "versus", "distinguish", "contrast"]
    if any(k in q_l for k in compare_keywords):
        return "compare"

    # ── Concept detection ─────────────────────────────────────────────────────
    concept_keywords = [
        "what is", "what are", "why is", "why are",
        "explain", "define", "definition",
        "how does", "describe", "purpose of", "pvt", "glass transition",
    ]
    if any(k in q_l for k in concept_keywords):
        return "concept"

    # ── Process / parameter detection ─────────────────────────────────────────
    process_keywords = [
        "processing temperature", "melt temperature", "mold temperature",
        "injection pressure", "holding pressure", "back pressure", "clamp force",
        "cooling time", "cycle time", "drying", "drying time", "drying temperature",
        "injection speed", "fill rate", "screw speed",
    ]
    if any(k in q_l for k in process_keywords):
        return "process"

    # ── List detection ────────────────────────────────────────────────────────
    # Note: 'causes' alone (without a defect keyword) maps to list, not defect
    list_keywords = [
        "list", "types of", "causes of", "factors", "enumerate",
        "what causes", "give me", "name the",
    ]
    if any(k in q_l for k in list_keywords):
        return "list"

    # ── Default ───────────────────────────────────────────────────────────────
    return "general"
