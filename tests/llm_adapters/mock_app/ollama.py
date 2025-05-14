from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import Literal, Optional
from . import REFERENCE_VECTOR

class EmbeddingRequest(BaseModel):
    model: Literal["mock_embedding"]
    input: list[str]
    truncate: Optional[bool] = False

router = APIRouter(tags=["ollama"])

@router.post("/api/embed")
async def embedding(request: EmbeddingRequest = Body(...)):
    return {
        "model": "mock_embedding",
        "embeddings": [REFERENCE_VECTOR for _ in request.input],
        "total_duration": 14143917,
        "load_duration": 1019500,
        "prompt_eval_count": 8
    }