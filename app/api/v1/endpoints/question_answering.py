
from fastapi import APIRouter, Depends, Body
from pydantic import BaseModel, Field
from typing import Dict

from app.services.qa_service import QAService

# Define the router
router = APIRouter()

# --- Data Models for Request/Response ---

class QARequest(BaseModel):
    question: str = Field(..., description="The question you want to ask.", example="ห้ามนายจ้างเรียกหลักประกันจากใคร")
    context: str = Field(..., description="The body of text where the answer should be found.", example="มาตรา 12 ห้ามมิให้นายจ้างเรียกหรือรับหลักประกันเพื่อการใดๆ จากลูกจ้าง เว้นแต่ลักษณะหรือสภาพของงานที่ทำนั้น ลูกจ้างต้องรับผิดชอบเกี่ยวกับการเงินหรือทรัพย์สินของนายจ้าง")

class QAResponse(BaseModel):
    score: float = Field(..., description="The model's confidence score (0.0 to 1.0).")
    answer: str = Field(..., description="The extracted answer.")

# --- Dependency Injection ---

# This function provides an instance of QAService to the route.
# FastAPI will cache this instance for the duration of the request.
def get_qa_service():
    return QAService()

# --- API Endpoint Definition ---

@router.post(
    "/ask",
    response_model=QAResponse,
    tags=["Question Answering"],
    summary="Ask a question based on a context",
    description="This endpoint uses a machine learning model to find the best answer to a question within the provided text (context)."
)
def ask_question(
    request_data: QARequest = Body(...),
    qa_service: QAService = Depends(get_qa_service)
) -> Dict:
    """
    Receives a question and a context, then returns the most likely answer found within the context.
    """
    result = qa_service.answer_question(
        question=request_data.question, 
        context=request_data.context
    )
    return result
