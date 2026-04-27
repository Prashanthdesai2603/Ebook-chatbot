from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.app.feedback_store import feedback_store

router = APIRouter()


class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    feedback: str  # "good" | "bad"


class FeedbackResponse(BaseModel):
    success: bool
    message: str


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    if request.feedback not in ("good", "bad"):
        raise HTTPException(status_code=400, detail="feedback must be 'good' or 'bad'")
    if not request.question.strip() or not request.answer.strip():
        raise HTTPException(status_code=400, detail="question and answer cannot be empty")

    feedback_store.save_feedback(
        session_id=request.session_id,
        question=request.question,
        answer=request.answer,
        feedback=request.feedback,
    )

    if request.feedback == "bad":
        print(
            f"[feedback] BAD response logged for session={request.session_id} "
            f"| question={request.question[:80]}"
        )

    return {"success": True, "message": "Thank you for your feedback!"}


@router.get("/feedback/analytics")
async def get_feedback_analytics():
    """Return counts of good vs bad responses."""
    return feedback_store.get_analytics()


@router.get("/feedback/good-examples")
async def get_good_examples(limit: int = 5):
    """Return top good Q&A examples for few-shot learning (internal use)."""
    examples = feedback_store.get_good_examples(limit=limit)
    return {"examples": examples}
