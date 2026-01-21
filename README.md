# Offline eBook Chatbot (RAG + LoRA)

A strictly offline, privacy-first AI chatbot that answers questions based **only** on a provided PDF ebook.
It uses **Retrieval Augmented Generation (RAG)** for factual accuracy and **LoRA (Low-Rank Adaptation)** for response styling.

## System Requirements
- OS: Windows 11
- CPU: Modern multi-core CPU (16GB RAM recommended)
- GPU: Not required (Runs purely on CPU)
- Python: 3.10+
- Node.js: 18+

## Setup Instructions

### 1. Installation
**Backend**:
```bash
cd backend
pip install -r requirements.txt
```

**Frontend**:
```bash
cd frontend
npm install
```

### 2. Data Preparation
1.  Place your **PDF ebook** inside `data/ebooks/`.
2.  Run the ingestion script to create the vector database:
    ```bash
    python backend/app/ingest.py
    ```

### 3. LoRA Training (Optional)
If you want to customize the **style** of the assistant (not facts):
1.  Prepare your dataset:
    ```bash
    python backend/lora/data_prep.py
    ```
    (Edit the generated `dataset.jsonl` with your stylistic examples).
2.  Train the adapter:
    ```bash
    python backend/lora/train.py
    ```
    The adapter will start loading automatically next time the backend starts.

## Running the Application

**Step 1: Start Backend**
```bash
cd backend
python -m app.main
```
(Server starts at http://localhost:8000)

**Step 2: Start Frontend**
```bash
cd frontend
npm run dev
```
(Client starts at http://localhost:5173 - opens in browser)

## Usage
- Open the frontend URL.
- Toggle between **Short** (concise) and **Detailed** (structured) modes.
- Ask questions. If the answer isn't in the book, the bot will say "I don't know based on the ebook."
