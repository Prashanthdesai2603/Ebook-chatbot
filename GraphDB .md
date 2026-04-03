Graph Database (Neo4j) Integration:
This chatbot uses a hybrid architecture combining:
Vector Database (Chroma) → for semantic search
Graph Database (Neo4j) → for relationship-based reasoning
The goal of adding a graph database is to improve answer completeness and technical accuracy, especially for engineering and defect-related queries.

Why Graph Database?
Traditional RAG (vector-only) systems:
  Retrieve relevant text
  Often return partial answers
Graph DB enhances this by:
  Capturing relationships between concepts
  Enabling multi-cause reasoning

Example:
Nylon → absorbs → Moisture  
Moisture → causes → Hydrolysis  
Hydrolysis → leads to → Splay  


   Architecture:
User Query
   ↓
Vector DB (Chroma) → Semantic Context
   +
Graph DB (Neo4j) → Relationships (cause, solution, material)
   ↓
Context Merger
   ↓
LLM (Gemini)
   ↓
Final Response


 Requirements:
1️. Software
Python 3.9+
Neo4j Desktop or Neo4j (Docker)
Existing backend setup

2️. Python Dependencies
Install required packages:
pip install neo4j python-dotenv


3️. Environment Variables
Update backend/.env:

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

Setup Instructions:
Step 1 — Start Neo4j
Open Neo4j Desktop
Create a local database
Start the instance

Verify:

http://localhost:7474
Step 2 — Insert Sample Data

Run in Neo4j browser:

MERGE (n:Material {name: "Nylon"})
MERGE (s:Issue {name: "Splay"})

MERGE (c1:Cause {name: "Moisture"})
MERGE (c2:Cause {name: "High Temperature"})
MERGE (c3:Cause {name: "High Injection Speed"})
MERGE (c4:Cause {name: "Material Degradation"})

MERGE (n)-[:HAS_ISSUE]->(s)

MERGE (s)-[:CAUSED_BY]->(c1)
MERGE (s)-[:CAUSED_BY]->(c2)
MERGE (s)-[:CAUSED_BY]->(c3)
MERGE (s)-[:CAUSED_BY]->(c4)
Step 3 — Verify Graph
MATCH (a)-[r]->(b) RETURN a,r,b;
🔧 Implementation Details
New File Added
backend/ai/graph_db.py

Responsible for:

Connecting to Neo4j
Querying relationships
Returning structured graph context
Modified File
backend/ai/rag_pipeline.py

Changes:

Integrated graph context retrieval
Merged vector + graph context before LLM call


How It Works:
1️.Query Processing
User query is analyzed to detect:
Defect (e.g., splay, burn)
Concept or process type

2️. Context Retrieval
Vector DB → retrieves relevant text chunks
Graph DB → retrieves causes, solutions, relationships

3️. Context Merging
Vector Knowledge + Graph Knowledge → Combined Context

4️. Response Generation
LLM generates answer using:
factual context (vector)
relational context (graph)