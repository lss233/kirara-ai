from pydantic import BaseModel, ConfigDict
from typing import cast, TypedDict, Literal, Optional

import requests
import asyncio

from kirara_ai.llm.adapter import LLMBackendAdapter, LLMEmbeddingProtocol, LLMReRankProtocol
from kirara_ai.llm.format.embedding import LLMEmbeddingRequest, LLMEmbeddingResponse
from kirara_ai.llm.format.rerank import LLMReRankRequest, LLMReRankResponse, ReRankerContent
from kirara_ai.llm.format.message import LLMChatTextContent, LLMChatImageContent
from kirara_ai.llm.format.response import Usage
from kirara_ai.media.manager import MediaManager
from kirara_ai.logger import get_logger

logger = get_logger("VoyageAdapter")

async def resolve_media_base64(inputs: list[LLMChatImageContent|LLMChatTextContent], media_manager: MediaManager) -> list:
    results = []
    for input in inputs:
        # voyage 的多模态接口设置中会将 一个content字段中的所有payload视作一个输入集，并对这个输入集合生成一个向量.
        # 所以这里对 image 做出处理，将其描述与原始图像打包为一个payload.
        if isinstance(input, LLMChatTextContent):
            results.append({
                "content": [
                    {"type": "text", "text": input.text}
                ]
            })
        elif isinstance(input, LLMChatImageContent):
            media = media_manager.get_media(input.media_id)
            if media is None:
                raise ValueError(f"Media {input.media_id} not found")
            results.append({
                "content": [
                    {"type": "text", "text": "" if (desc := media.description) is None else desc},
                    {"type": "image_base64", "image_base64": await media.get_base64()}
                ]
            })
    return results

class ReRankData(TypedDict):
    index: int
    relevance_score: float
    document: Optional[str]

class ReRankResponse(TypedDict):
    """给mypy检查用, 顺便给开发者标识返回json的基本结构。"""
    object: Literal["list"]
    data: list[ReRankData]
    model: str
    usage: dict[Literal["total_tokens"], int]

class EmbeddingData(TypedDict):
    object: Literal["embedding"]
    embedding: list[float | int]
    index: int

class EmbeddingResponse(TypedDict):
    object: Literal["list"]
    data: list[EmbeddingData]
    model: str
    usage: dict[Literal["total_tokens"], int]

class ModalEmbeddingResponse(TypedDict):
    object: Literal["list"]
    data: list[EmbeddingData]
    model: str
    # voyage 的多模态接口会返回三个usage指标: text_tokens: 文字使用token数, image_pixels: 图片像素数, total_tokens: 总token数
    usage: dict[Literal["text_tokens", "image_pixels", "total_tokens"], int]

class VoyageConfig(BaseModel):
    api_key: str
    api_base: str = "https://api.voyageai.com"
    model_config = ConfigDict(frozen=True)

class VoyageAdapter(LLMBackendAdapter, LLMEmbeddingProtocol, LLMReRankProtocol):
    media_manager: MediaManager

    def __init__(self, config: VoyageConfig):
        self.config = config
    
    def embed(self, req: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        # voyage 支持多模态嵌入, 但是两个接口支持的参数不同。
        # 因此对其做区分，以充分利用 voyage 接口提供的可选参数。
        if any(isinstance(input, LLMChatImageContent) for input in req.inputs):
            return self._multi_modal_embedding(req)
        else:
            return self._text_embedding(req)
        
    def _text_embedding(self, req: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        api_url = f"{self.config.api_base}/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        inputs = cast(list[LLMChatTextContent], req.inputs)
        data = {
            "model": req.model,
            "input": [input.text for input in inputs],
            "truncation": req.truncate,
            "input_type": req.input_type,
            "output_dimension": req.dimension,
            "output_dtype": req.encoding_format,
            "encoding_format": req.encoding_format,
        }
        data = { k:v for k,v in data.items() if v is not None }

        response = requests.post(api_url, headers=headers, json=data)
        try:
            response.raise_for_status()
            response_data: EmbeddingResponse = response.json()
        except Exception as e:
            logger.error(f"Response: {response.text}")
            raise e
        
        return LLMEmbeddingResponse(
            vectors=[data["embedding"] for data in response_data["data"]],
            usage = Usage(
                total_tokens=response_data["usage"].get("total_tokens", 0)
            )
        )
    
    def _multi_modal_embedding(self, req: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        api_url = f"{self.config.api_base}/v1/multimodalembeddings"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        # loop = asyncio.new_event_loop()
        # try:
        #     asyncio.set_event_loop(loop)
        #     data = {
        #         "model": req.model,
        #         "inputs": loop.run_until_complete(resolve_media_base64(req.inputs, self.media_manager)),
        #         "input_type": req.input_type,
        #         "truncation": req.truncate,
        #         "output_encoding": req.encoding_format
        #     }
        # finally:
        #     loop.close() # 关闭事件循环，避免资源泄露。
        #     asyncio.set_event_loop(None) # 解除 asyncio 事件循环绑定。避免get_running_loop()获取到已结束时间循环。

        data = {
            "model": req.model,
            # 为何不使用神奇的 asyncio.run() 自动管理这个临时loop的生命周期呢。（python 3.7+）
            "inputs": asyncio.run(resolve_media_base64(req.inputs, self.media_manager)),
            "input_type": req.input_type,
            "truncation": req.truncate,
            "output_encoding": req.encoding_format
        }
        data = { k:v for k,v in data.items() if v is not None }

        response = requests.post(api_url, headers=headers, json=data)
        try:
            response.raise_for_status()
            response_data: ModalEmbeddingResponse = response.json()
        except Exception as e:
            logger.error(f"Response: {response.text}")
            raise e
        
        return LLMEmbeddingResponse(
            vectors=[data["embedding"] for data in response_data["data"]],
            usage = Usage(
                total_tokens=response_data["usage"].get("total_tokens", 0)
            )
        )

    def rerank(self, req: LLMReRankRequest) -> LLMReRankResponse:
        api_url = f"{self.config.api_base}/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "query": req.query,
            "documents": req.documents,
            "model": req.model,
            "top_k": req.top_k,
            "return_documents": req.return_documents,
            "truncation": req.truncation
        }

        # 去除 None 值
        data = { k:v for k,v in data.items() if v is not None }

        response = requests.post(api_url, headers=headers, json=data)
        try:
            response.raise_for_status()
            response_data: ReRankResponse = response.json()
            logger.debug(f"server  response_data: {response_data}")
        except Exception as e:
            logger.error(f"Response: {response.text}")
            raise e
        
        return LLMReRankResponse(
            contents = [ReRankerContent(
                    document = data.get("document", None),
                    score = data["relevance_score"]
            ) for data in response_data["data"]],
            usage = Usage(
                total_tokens = response_data["usage"].get("total_tokens", 0)
            ),
            sort = cast(bool, req.sort) # 强制类型转换，避免mypy报错。
        )
        