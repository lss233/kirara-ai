from abc import ABC
from typing import Protocol, runtime_checkable

from kirara_ai.llm.format.request import LLMChatRequest
from kirara_ai.llm.format.response import LLMChatResponse
from kirara_ai.llm.format.embedding import LLMEmbeddingRequest, LLMEmbeddingResponse
from kirara_ai.llm.format.rerank import LLMReRankRequest, LLMReRankResponse
from kirara_ai.media.manager import MediaManager
from kirara_ai.tracing.llm_tracer import LLMTracer


@runtime_checkable
class AutoDetectModelsProtocol(Protocol):
    async def auto_detect_models(self) -> list[str]: ...

@runtime_checkable
class LLMChatProtocol(Protocol):
    def chat(self, req: LLMChatRequest) -> LLMChatResponse: ...

@runtime_checkable
class LLMEmbeddingProtocol(Protocol):
    def embed(self, req: LLMEmbeddingRequest) -> LLMEmbeddingResponse: ...

@runtime_checkable
class LLMReRankProtocol(Protocol):
    def rerank(self, req: LLMReRankRequest) -> LLMReRankResponse: ...

class LLMBackendAdapter(ABC):
    backend_name: str
    media_manager: MediaManager
    tracer: LLMTracer