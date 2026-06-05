eBook Chatbot:
privacy-first AI chatbot that answers questions based only on a provided PDF ebook. It uses Retrieval Augmented Generation (RAG) for factual accuracy and LoRA (Low-Rank Adaptation) for response styling.

System Requirements
OS: Windows 11
CPU: Modern multi-core CPU (16GB RAM recommended)
GPU: Not required (Runs purely on CPU)
Python: 3.10+
Node.js: 18+
Setup Instructions
1. Installation
Backend:

cd backend
pip install -r requirements.txt
python -m app.main
Frontend:

cd frontend
npm install
2. Data Preparation
Place your PDF ebook inside data/ebooks/.
Run the ingestion script to create the vector database:
python app/ingest.py
3. LoRA Training (Optional)
If you want to customize the style of the assistant (not facts):

Prepare your dataset:
python lora/data_prep.py
(Edit the generated dataset.jsonl with your stylistic examples).
Train the adapter:
python lora/train.py
The adapter will start loading automatically next time the backend starts.
Running the Application
Step 1: Start Backend

cd backend
python -m app.main
(Server starts at http://localhost:8000)

Step 2: Start Frontend

cd frontend
npm run dev
(Client starts at http://localhost:5173 - opens in browser)

Usage
Open the frontend URL.
Toggle between Short (concise) and Detailed (structured) modes.
Ask questions. If the answer isn't in the book, the bot will say "I don't know based on the ebook."