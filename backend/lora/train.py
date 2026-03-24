import os
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# Config
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "lora_adapters"
DATA_FILE = BASE_DIR / "backend" / "lora" / "dataset.jsonl" # Assuming dataset is generated in lora dir

def train_lora():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="cpu", 
        torch_dtype=torch.float32 
    )

    # LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )
    
    model = get_peft_model(model, peft_config)
    
    print("Printing trainable parameters:")
    model.print_trainable_parameters()

    # Load Dataset
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found. Please create it first.")
        return

    print("Loading dataset...")
    # JSONL format: {"text": "instruction... response..."}
    dataset = load_dataset("json", data_files=str(DATA_FILE), split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=252)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1, # Low batch size for CPU
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        use_cpu=True 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("Starting Training...")
    trainer.train()

    print("Saving adapter...")
    model.save_pretrained(str(OUTPUT_DIR))
    print("Done!")

if __name__ == "__main__":
    train_lora()
