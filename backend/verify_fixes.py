import sys
import os

# Add backend directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.rag import rag_pipeline

def run_test(question, mode="short", expected_type="answer"):
    print(f"\n--- Testing: '{question}' [Mode: {mode}] ---")
    response = rag_pipeline.answer_query(question, mode=mode)
    print(f"Response:\n{response}")
    
    if expected_type == "refusal":
        if "I don't know based on the ebook" in response:
            print("✅ PASS: Correctly refused.")
        else:
            print("❌ FAIL: Did not refuse as expected.")
    elif expected_type == "answer":
        if "I don't know based on the ebook" not in response and len(response) > 20:
            print("✅ PASS: Provided an answer.")
        else:
            print("❌ FAIL: Refused or provided empty answer for valid question.")
            
    if mode == "detailed" and expected_type == "answer":
        if "-" in response or "*" in response:
             print("✅ PASS: Output contains bullet points (likely detailed).")
        else:
             print("⚠️ WARN: Detailed mode might lack bullet points.")

if __name__ == "__main__":
    print("Testing eBook Chatbot Fixes...")
    
    # 1. Out of Ebook (Hard Guardrails)
    run_test("Who is Rohit Sharma?", expected_type="refusal")
    run_test("What is the capital of France?", expected_type="refusal")
    
    # 2. In Ebook (Retrieval + Answer)
    # Assuming these concepts exist in the book based on User's prompt examples
    run_test("What are the three types of consistencies required in injection molding?", mode="short", expected_type="answer")
    run_test("What are the five critical factors of molding?", mode="detailed", expected_type="answer")
    
    # 3. Mode Checks
    run_test("Explain the cooling phase.", mode="detailed", expected_type="answer")
    
    print("\nTests Completed.")
