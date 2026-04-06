import json
import os
from pathlib import Path

# Paths to knowledge graph files
BASE_DIR = Path(__file__).resolve().parent.parent.parent   # backend/ai/ -> backend/ -> project root
KG_DIR = BASE_DIR / "data" / "knowledge_graph"


def load_json(filename):
    path = KG_DIR / filename
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}

def query_knowledge_graph(question: str):
    """
    Queries JSON knowledge files and detects if question relates to:
    - defect
    - material
    - process concept
    Returns relevant structured information.
    """
    question_l = question.lower()
    
    # Load data
    defects: dict = load_json("defects.json")
    materials: dict = load_json("materials.json")
    process_terms: dict = load_json("process_terms.json")
    
    results: dict = {
        "defects": [],
        "materials": [],
        "concepts": []
    }
    
    # Check for defects
    if isinstance(defects, dict):
        for defect, data in defects.items():
            if defect in question_l:
                results["defects"].append({defect: data})
                
    # Check for materials
    if isinstance(materials, dict):
        for material, data in materials.items():
            if material.lower() in question_l:
                results["materials"].append({material: data})
                
    # Check for process concepts
    if isinstance(process_terms, dict):
        for term, definition in process_terms.items():
            if term in question_l:
                results["concepts"].append({term: definition})
            
    # Format the output for context merging
    output = []
    if results["defects"]:
        output.append("--- DEFECT DATA ---")
        for d in results["defects"]:
            output.append(json.dumps(d, indent=2))
            
    if results["materials"]:
        output.append("--- MATERIAL DATA ---")
        for m in results["materials"]:
            output.append(json.dumps(m, indent=2))
            
    if results["concepts"]:
        output.append("--- PROCESS CONCEPTS ---")
        for c in results["concepts"]:
            output.append(json.dumps(c, indent=2))
            
    return "\n".join(output) if output else "No specific engineering knowledge found in graph."
