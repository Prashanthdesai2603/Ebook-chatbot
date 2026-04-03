import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

class GraphDB:
    def __init__(self):
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            # Basic connectivity check
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"Connected to Neo4j at {NEO4J_URI}")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def get_causes(self, issue: str):
        if not self.driver:
            return []

        query = """
        MATCH (i:Issue {name:$issue})-[:CAUSED_BY]->(c:Cause)
        RETURN c.name AS cause
        """
        causes = []
        try:
            with self.driver.session() as session:
                result = session.run(query, issue=issue)
                for record in result:
                    causes.append(record["cause"])
        except Exception as e:
            print(f"Neo4j query error: {e}")
        
        return causes

def get_graph_context(query: str) -> str:
    """
    Returns a formatted, labeled cause list from the Neo4j graph database
    for the detected defect type. Returns an empty string if no match.
    """
    query_lower = query.lower()

    # Map query keywords -> Neo4j Issue node name
    defect_map = {
        "splay":         "Splay",
        "silver streak": "Splay",
        "burn":          "Burn",
        "burning":       "Burn",
        "warpage":       "Warpage",
        "warp":          "Warpage",
        "sink mark":     "Sink Mark",
        "sinking":       "Sink Mark",
        "short shot":    "Short Shot",
        "short fill":    "Short Shot",
        "flash":         "Flash",
        "weld line":     "Weld Line",
        "knit line":     "Weld Line",
        "void":          "Voids",
        "bubble":        "Voids",
        "discolor":      "Discoloration",
        "black spot":    "Discoloration",
        "jetting":       "Jetting",
        "flow mark":     "Flow Marks",
        "flow line":     "Flow Marks",
    }

    issue = None
    for keyword, node_name in defect_map.items():
        if keyword in query_lower:
            issue = node_name
            break

    if not issue:
        return ""

    db = GraphDB()
    if not db.driver:
        return ""

    try:
        causes = db.get_causes(issue)
        if not causes:
            return ""

        # Return as labeled bullet list for LLM clarity
        causes_formatted = "\n".join(f"• {c}" for c in causes)
        return f"Graph-identified causes for '{issue}':\n{causes_formatted}"
    except Exception as e:
        print(f"Error getting graph context: {e}")
        return ""
    finally:
        db.close()

# Example usage for testing
if __name__ == "__main__":
    test_query = "What causes splay in injection molding?"
    print(f"Query: {test_query}")
    print("Graph Context:")
    print(get_graph_context(test_query))
