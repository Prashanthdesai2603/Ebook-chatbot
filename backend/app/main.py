from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
from pathlib import Path

# Add project root to sys.path to allow importing from backend.ai/ directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.ai.rag_pipeline import rag_pipeline
from backend.app.mysql_logger import chat_logger
from backend.app.routes import auth

app = FastAPI(title="Offline eBook Chatbot")

origins = [
    "https://chatbot.fimmtech.com",
]

if os.getenv("APP_ENV") == "development":
    origins.append("http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Authentication Routes
app.include_router(auth.router, prefix="/api", tags=["Auth"])


class ChatRequest(BaseModel):
    message: str
    session_id: str = None
    mode: str = "short"  # "short" or "detailed"

session_store = {}

class ChatResponse(BaseModel):
    response: str

@app.get("/")
def read_root():
    return {"status": "online", "system": "Offline eBook Chatbot"}

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Empty message")
            
        session_id = request.session_id or "default_session"
        print(f"Received query: {request.message} [{request.mode}] for session: {session_id}")
        
        # 1. Fetch Chat History from Memory
        history = session_store.get(session_id, [])
        
        # 2. Format History for Prompt
        history_context = ""
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_context += f"{role}: {msg['content']}\n"
        
        # 3. Generate Answer with History context
        answer = rag_pipeline.answer_query(request.message, mode=request.mode, history_context=history_context)
        
        # 4. Store messages in memory
        if session_id not in session_store:
            session_store[session_id] = []
        
        session_store[session_id].append({"role": "user", "content": request.message})
        session_store[session_id].append({"role": "assistant", "content": answer})
        
        # 5. Limit memory size (Keep only last 10 messages)
        session_store[session_id] = session_store[session_id][-10:]
        
        # 6. Save to MySQL (Optional Logging)
        chat_logger.save_chat(session_id, request.message, answer)
        
        return {"response": answer}
        
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
