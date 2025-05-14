from fastapi import APIRouter, Body
from pydantic import BaseModel

from typing import Literal, Union, Optional
from . import REFERENCE_VECTOR

class EmbeddingRequest(BaseModel):
    input: list[str]
    model: Literal["mock_embedding"]

class ReRankRequest(BaseModel):
    query: str
    documents: list[str]
    model: Literal["mock_rerank"]
    return_documents: Optional[bool] = False

class TextContent(BaseModel):
    type: Literal["text"]
    text: str

class ImageBase64Content(BaseModel):
    type: Literal["image_base64"]
    image_base64: str

class ImageUrlContent(BaseModel):
    type: Literal["image_url"]
    image_url: str

class CombinedContent(BaseModel):
    content: list[Union[TextContent, ImageBase64Content, ImageUrlContent]]
    
class MultiModalRequest(BaseModel):
    inputs: list[CombinedContent]
    model: Literal["mock_multimodal"]

router = APIRouter(tags=["voyage"])

@router.post("/v1/embeddings")
async def get_embeddings(request: EmbeddingRequest = Body(...)):
    return {
        "object": "list",
        "data": [
            {
            "object": "embedding",
            "embedding": REFERENCE_VECTOR, # 使用固定的向量列表方便验证
            "index": 0
            }
        ],
        "model": "mock_embedding",
        "usage": {
            "total_tokens": 10
        }
    }

@router.post("/v1/multimodalembeddings")
async def get_multimodal_embeddings(request: MultiModalRequest = Body(...)):
    return {
        "object": "list",
        "data": [
            {
            "object": "embedding",
            "embedding": REFERENCE_VECTOR, # 使用固定的向量列表方便验证
            "index": 0
            }
        ],
        "model": "mock_multimodal",
        "usage": {
            "text_tokens": 5,
            "image_pixels": 2000000,
            "total_tokens": 3576
        }
    }

@router.post("/v1/rerank")
async def get_rerank(request: ReRankRequest = Body(...)):
    print(request)
    return {
        "object": "list",
        "data": [
            {
            "index": 0,
            "relevance_score": 0.4375,
            "document": request.documents[0],
            },
            {
            "index": 1,
            "relevance_score": 0.421875,
            "document": request.documents[1],
            }
        ],
        "model": "mock_rerank",
        "usage": {
            "total_tokens": 26
        }
    }