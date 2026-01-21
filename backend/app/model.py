import torch
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Configuration
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Path to local adapter if exists
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LORA_ADAPTER_DIR = BASE_DIR / "data" / "lora_adapters"

class LLMEngine:
    def __init__(self):
        print("Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        
        print("Loading Base Model (CPU)...")
        # Load in fp32 or bf16 if supported, but usually fp32 is safest for pure CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map="cpu", 
            torch_dtype=torch.float32 
        )
        
        # Check for LoRA adapter
        abs_adapter_dir = str(LORA_ADAPTER_DIR.resolve())
        if LORA_ADAPTER_DIR.exists() and os.listdir(abs_adapter_dir):
            print(f"Found LoRA adapter at {abs_adapter_dir}. Loading...")
            try:
                self.model = PeftModel.from_pretrained(self.model, abs_adapter_dir)
                print("LoRA Adapter loaded successfully.")
            except Exception as e:
                print(f"Failed to load LoRA adapter: {e}")
        else:
            print("No LoRA adapter found. Using base model.")

        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9
            )
            
        # Slice the new tokens only (exclude the input prompt tokens)
        generated_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()

llm_engine = LLMEngine()
