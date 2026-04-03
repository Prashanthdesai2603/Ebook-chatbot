from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Load env
load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

print("Connecting with:")
print("URI:", URI)
print("USER:", USER)
print("PASS:", PASSWORD)

try:
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    with driver.session() as session:
        result = session.run("MATCH (n) RETURN n LIMIT 5")
        print("\n✅ SUCCESS: Connected to Neo4j")
        for record in result:
            print(record)

except Exception as e:
    print("\n❌ FAILED:", e)