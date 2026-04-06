import os
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

# File is now at backend/ai/build_graph.py
# .parent      = backend/ai/
# .parent.parent = backend/    <-- where .env lives
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j123")

print(f"Connecting to Neo4j at {URI} as '{USER}'...")
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


# ──────────────────────────────────────────────────────────────
# Low-level helper: single MERGE insert using the module driver
# ──────────────────────────────────────────────────────────────
def insert_graph(material, issue, causes, solutions):
    query = """
    MERGE (m:Material {name:$material})
    MERGE (i:Issue {name:$issue})
    MERGE (m)-[:HAS_ISSUE]->(i)

    FOREACH (c IN $causes |
        MERGE (cause:Cause {name:c})
        MERGE (i)-[:CAUSED_BY]->(cause)
    )

    FOREACH (s IN $solutions |
        MERGE (sol:Solution {name:s})
        MERGE (i)-[:HAS_SOLUTION]->(sol)
    )
    """
    with driver.session() as session:
        session.run(query, material=material, issue=issue, causes=causes, solutions=solutions)


# ──────────────────────────────────────────────────────────────
# Full ebook graph using GraphDB class (relative import)
# ──────────────────────────────────────────────────────────────
def insert_full_ebook_graph():
    from backend.ai.graph_db import GraphDB   # absolute import works since ROOT is on sys.path
    db = GraphDB()

    data = [
        # --- SPLAY ---
        {
            "material": "Nylon",
            "issue": "Splay",
            "causes": [
                "Moisture in resin",
                "Excessive melt temperature",
                "High injection speed",
                "Material degradation"
            ],
            "solutions": [
                "Proper drying of material",
                "Reduce melt temperature",
                "Optimize injection speed"
            ]
        },
        # --- BURN MARKS ---
        {
            "material": "ABS",
            "issue": "Burn Marks",
            "causes": [
                "Trapped air",
                "High injection speed",
                "Poor venting"
            ],
            "solutions": [
                "Improve venting",
                "Reduce injection speed",
                "Optimize mold design"
            ]
        },
        # --- WARPAGE ---
        {
            "material": "PP",
            "issue": "Warpage",
            "causes": [
                "Uneven cooling",
                "Material shrinkage variation",
                "Improper mold temperature"
            ],
            "solutions": [
                "Uniform cooling",
                "Optimize mold temperature",
                "Balanced mold design"
            ]
        },
        # --- SINK MARK ---
        {
            "material": "PC",
            "issue": "Sink Mark",
            "causes": [
                "Thick sections",
                "Low packing pressure",
                "Insufficient cooling"
            ],
            "solutions": [
                "Increase packing pressure",
                "Reduce thickness",
                "Improve cooling"
            ]
        },
        # --- SHORT SHOT ---
        {
            "material": "ABS",
            "issue": "Short Shot",
            "causes": [
                "Low injection pressure",
                "Low melt temperature",
                "Poor flow"
            ],
            "solutions": [
                "Increase injection pressure",
                "Increase melt temperature",
                "Improve flow design"
            ]
        },
        # --- CONCEPT: PVT ---
        {
            "material": "Polymer",
            "issue": "PVT Behavior",
            "causes": [
                "Temperature change",
                "Pressure variation"
            ],
            "solutions": [
                "Control cooling curve",
                "Maintain linear shrinkage",
                "Optimize process consistency"
            ]
        },
        # --- COLOR VARIATION ---
        {
            "material": "Plastic",
            "issue": "Color Variation",
            "causes": [
                "Improper mixing",
                "Incompatible colorants",
                "Separation of pigments"
            ],
            "solutions": [
                "Uniform mixing",
                "Use compatible carriers",
                "Control processing conditions"
            ]
        }
    ]

    with db.driver.session() as session:
        for item in data:
            session.run("""
                MERGE (m:Material {name: $material})
                MERGE (i:Issue {name: $issue})
                MERGE (m)-[:HAS_ISSUE]->(i)
                WITH i
                UNWIND $causes AS cause
                MERGE (c:Cause {name: cause})
                MERGE (i)-[:CAUSED_BY]->(c)
                WITH i
                UNWIND $solutions AS solution
                MERGE (s:Solution {name: solution})
                MERGE (i)-[:HAS_SOLUTION]->(s)
            """,
            material=item["material"],
            issue=item["issue"],
            causes=item["causes"],
            solutions=item["solutions"]
            )

    db.close()
    print("✅ FULL EBOOK GRAPH INSERTED")


# ──────────────────────────────────────────────────────────────
# build_sample: inserts curated sample data using module driver
# ──────────────────────────────────────────────────────────────
def build_sample():
    insert_graph(
        "Nylon", "Splay",
        ["Moisture", "High Temperature", "Injection Speed", "Material Degradation"],
        ["Proper Drying", "Reduce Temperature", "Optimize Speed"]
    )
    insert_graph(
        "ABS", "Burn Marks",
        ["Trapped Air", "High Injection Speed", "Poor Venting"],
        ["Improve Venting", "Reduce Speed", "Lower Temperature"]
    )
    insert_graph(
        "Polypropylene", "Warpage",
        ["Uneven Cooling", "Mold Temperature Variation", "Residual Stress", "High Packing Pressure"],
        ["Uniform Cooling", "Optimize Mold Design"]
    )
    insert_graph(
        "ABS", "Sink Marks",
        ["Low Packing Pressure", "Thick Sections", "High Temperature", "Short Hold Time"],
        ["Increase Packing Pressure", "Optimize Cooling"]
    )
    insert_graph(
        "Nylon", "Splay",
        ["Moisture Content", "High Temperature", "Injection Speed", "Material Degradation"],
        ["Proper Drying", "Reduce Temperature", "Optimize Speed"]
    )
    insert_graph(
        "Polycarbonate", "Flash",
        ["High Injection Pressure", "Mold Cavity Imbalance", "Poor Clamping Force"],
        ["Reduce Pressure", "Improve Mold Design", "Increase Clamping Force"]
    )
    insert_graph(
        "Polypropylene", "Short Shot",
        ["Low Melt Temperature", "Insufficient Packing Time", "Material Degradation"],
        ["Increase Temperature", "Extend Hold Time", "Use Virgin Material"]
    )
    insert_graph(
        "PP", "Warpage",
        ["Uneven Cooling", "High Mold Temperature", "Thick Wall Section", "High Packing Pressure"],
        ["Optimize Cooling Channels", "Reduce Mold Temperature", "Uniform Wall Thickness"]
    )
    insert_graph(
        "PC", "Sink Mark",
        ["Insufficient Packing Pressure", "Short Holding Time", "Thick Wall Section", "High Melt Temperature"],
        ["Increase Holding Pressure", "Extend Hold Time", "Reduce Wall Thickness"]
    )


if __name__ == "__main__":
    build_sample()
    print("Graph data inserted successfully.")
    driver.close()