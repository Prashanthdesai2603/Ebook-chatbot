import json
import os
from pathlib import Path

def create_sample_dataset():
    """
    Creates a sample JSONL dataset for LoRA training.
    """
    data = [
        {"text": "<|system|>\nAnswer strictly from the book.\n</s>\n<|user|>\nSummarize the introduction.\n</s>\n<|assistant|>\nThe introduction covers the basic principles of AI.\n</s>"},
        {"text": "<|system|>\nAnswer strictly from the book.\n</s>\n<|user|>\nWhat is RAG?\n</s>\n<|assistant|>\nRAG stands for Retrieval Augmented Generation.\n</s>"}
    ]
    
    BASE_DIR = Path(__file__).resolve().parent
    output_file = BASE_DIR / "dataset.jsonl"
    with open(output_file, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")
            
    print(f"Created sample {output_file}. Please populate this with real data from your ebook for best results.")

if __name__ == "__main__":
    create_sample_dataset()
