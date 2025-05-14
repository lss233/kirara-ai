from fastapi import APIRouter, Body, Query, Depends
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
from typing import Literal
from . import REFERENCE_VECTOR
from .app import AUTH_KEY

class BatchEmbeddingPart(BaseModel):
    text: str
class BatchEmbeddingParts(BaseModel):
    parts: list[BatchEmbeddingPart]
class BatchEmbeddingPayload(BaseModel):
    model: Literal["mock_embedding"]
    content: BatchEmbeddingParts

class BatchEmbeddingRequest(BaseModel):
    requests: list[BatchEmbeddingPayload]

async def gemini_authenticate(key: str = Query(...)):
    if key != AUTH_KEY:
        raise HTTPException(status_code=401, detail="Invalid authentication key")

router = APIRouter(tags=["gemini"], dependencies=[Depends(gemini_authenticate)])

@router.post("/models/{model}:batchEmbedContents")
async def batch_embed_contents(model: str, request: BatchEmbeddingRequest = Body(...)):
    return {
        "embeddings": [
            {
                "values": REFERENCE_VECTOR
            }
        ]
    }