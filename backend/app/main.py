# backend/app/main.py
# sys.path manipulation removed — PYTHONPATH=/app is set in the Dockerfile
# and in docker-compose, so `from ai.rag_pipeline import ...` resolves cleanly.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend.ai.rag_pipeline import rag_pipeline
from backend.app.mysql_logger import chat_logger

app = FastAPI(title="Offline eBook Chatbot")

# CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    mode: str = "short"  # "short" or "detailed"

class ChatResponse(BaseModel):
    response: str

@app.get("/")
def read_root():
    return {"status": "online", "system": "Offline eBook Chatbot"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Empty message")
            
        print(f"Received query: {request.message} [{request.mode}]")
        answer = rag_pipeline.answer_query(request.message, mode=request.mode)
        
        # Log to MySQL (Optional task 7)
        chat_logger.log_chat(request.message, answer)
        
        return {"response": answer}
        
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
