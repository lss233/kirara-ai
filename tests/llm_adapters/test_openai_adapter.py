from kirara_ai.plugins.llm_preset_adapters.openai_adapter import OpenAIAdapter, OpenAIConfig
from kirara_ai.llm.format.embedding import LLMEmbeddingRequest, LLMEmbeddingResponse
from kirara_ai.llm.format.message import LLMChatTextContent, LLMChatImageContent, LLMChatMessage
from kirara_ai.llm.format.request import LLMChatRequest
from kirara_ai.llm.format.response import LLMChatResponse

from unittest.mock import patch
import pytest

from .mock_app import AUTH_KEY, REFERENCE_VECTOR, OPENAI_ENDPOINT
class TestOpenAIAdapter:
    @pytest.fixture(scope="class")
    def openai_adapter(self, mock_media_manager) -> OpenAIAdapter:
        # 能力有限没法在这里把patch整合进来
        config  = OpenAIConfig(
            api_base=OPENAI_ENDPOINT,
            api_key=AUTH_KEY
        )
        adapter = OpenAIAdapter(config)
        adapter.media_manager = mock_media_manager

        return adapter

    def test_embed(self, openai_adapter: OpenAIAdapter):
        req = LLMEmbeddingRequest(
            inputs=[LLMChatTextContent(text="hello world", type="text")],
            model="mock_embedding",
        )

        response = openai_adapter.embed(req)
        assert isinstance(response, LLMEmbeddingResponse)
        assert response.vectors[0] == REFERENCE_VECTOR

    def test_embed_with_image(self, openai_adapter: OpenAIAdapter):
        req = LLMEmbeddingRequest(
            inputs=[LLMChatImageContent(media_id="mock_media_id", type="image")],
            model="mock_embedding",
        )

        with pytest.raises(ValueError, match="openai does not support multi-modal embedding"):
            openai_adapter.embed(req)

    def test_embed_with_input_out_of_range(self, openai_adapter: OpenAIAdapter):
        req = LLMEmbeddingRequest(
            inputs=[LLMChatTextContent(text="hello world", type="text") for _ in range(2050)],
            model="mock_embedding"
        )
        
        with pytest.raises(ValueError, match="Text list has too many dimensions, max dimension is 2048"):
            openai_adapter.embed(req)

    def test_old_embedding_model_rasies_error(self, openai_adapter: OpenAIAdapter):
        req = LLMEmbeddingRequest(
            model = "text-embedding-ada-002",
            inputs=[LLMChatTextContent(text="hello world", type="text")],
            dimension=512,
        )

        from requests.exceptions import HTTPError
        with pytest.raises(HTTPError):
            openai_adapter.embed(req)

    # patch不生效，跳过该项测试相信后人的智慧
    @pytest.mark.skip(reason="patch装饰器未生效，该项测试完全无法进行")
    def test_normal_chat(self, openai_adapter: OpenAIAdapter):
        with patch("kirara_ai.tracing.trace_llm_chat", lambda func: func):
            req = LLMChatRequest(
                messages=[
                    LLMChatMessage(
                        role="system",
                        content=[
                            LLMChatTextContent(text="你是一个猫娘。"),
                            LLMChatTextContent(text="hello world")
                        ]
                    ),
                ],
                model="mock_chat",
            )

            response = openai_adapter.chat(req)
            assert isinstance(response, LLMChatResponse)
            assert response.message.content[0].text == "mock_response"
            assert response.message.role == "assistant"
            assert response.message.tool_calls is None
            assert response.usage.total_tokens == 29 #type: ignore