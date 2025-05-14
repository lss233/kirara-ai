from kirara_ai.plugins.llm_preset_adapters.ollama_adapter import OllamaAdapter, OllamaConfig
from kirara_ai.llm.format.message import LLMChatTextContent, LLMChatImageContent
from kirara_ai.llm.format.embedding import LLMEmbeddingRequest, LLMEmbeddingResponse

import pytest
from .mock_app import REFERENCE_VECTOR, OLLAMA_ENDPOINT

class TestOllamaAdapter:
    @pytest.fixture(scope="class")
    def ollama_adapter(self, mock_media_manager) -> OllamaAdapter:
        config = OllamaConfig(
            api_base=OLLAMA_ENDPOINT,
        )
        adapter = OllamaAdapter(config)
        adapter.media_manager = mock_media_manager
        return adapter
    
    def test_embedding(self, ollama_adapter: OllamaAdapter):
        req = LLMEmbeddingRequest(
            inputs=[LLMChatTextContent(text="hello world", type="text")],
            model="mock_embedding"
        )

        response = ollama_adapter.embed(req)
        assert isinstance(response, LLMEmbeddingResponse)
        assert response.vectors[0] == REFERENCE_VECTOR

    def test_embedding_with_image(self, ollama_adapter: OllamaAdapter):
        req = LLMEmbeddingRequest(
            inputs=[LLMChatImageContent(media_id="1234567890", type="image")],
            model="mock_embedding"
        )

        # 检测其是否会检出不支持的图片类型。目前ollama嵌入不支持多模态
        with pytest.raises(ValueError, match="ollama api does not support multi-modal embedding"):
            ollama_adapter.embed(req)