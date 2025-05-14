from kirara_ai.plugins.llm_preset_adapters.gemini_adapter import GeminiAdapter, GeminiConfig
from kirara_ai.llm.format.embedding import LLMEmbeddingRequest, LLMEmbeddingResponse
from kirara_ai.llm.format.message import LLMChatTextContent, LLMChatImageContent

import pytest

from .mock_app import AUTH_KEY, GEMINI_ENDPOINT, REFERENCE_VECTOR

class TestGeminiAdapter:
    @pytest.fixture(scope="class")
    def gemini_adapter(self, mock_media_manager) -> GeminiAdapter:
        config = GeminiConfig(
            api_key=AUTH_KEY,
            api_base=GEMINI_ENDPOINT
        )
        adapter = GeminiAdapter(config)
        adapter.media_manager = mock_media_manager
        return adapter

    @pytest.mark.skip(reason="暂未实现测试集")
    def test_chat(self, gemini_adapter):
        # 如果你觉得想测试这个可以添加一些测试用例
        pass

    def test_embed(self, gemini_adapter: GeminiAdapter):
        req = LLMEmbeddingRequest(
            inputs=[
                LLMChatTextContent(text="hello world", type="text")
            ],
            model = "mock_embedding"
        )
        
        response = gemini_adapter.embed(req)
        assert isinstance(response, LLMEmbeddingResponse)
        assert response.vectors[0] == REFERENCE_VECTOR
