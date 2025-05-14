from kirara_ai.plugins.llm_preset_adapters.voyage_adapter import VoyageAdapter, VoyageConfig
from kirara_ai.llm.format.embedding import LLMEmbeddingRequest, LLMEmbeddingResponse
from kirara_ai.llm.format.rerank import LLMReRankRequest, LLMReRankResponse
from kirara_ai.llm.format.message import LLMChatTextContent, LLMChatImageContent

import pytest

from .mock_app import VOYAGE_ENDPOINT, AUTH_KEY, REFERENCE_VECTOR

class TestVoyageAdapter:
    @pytest.fixture(scope="class")
    def voyage_adapter(self, mock_media_manager):
        config = VoyageConfig(
            api_base=VOYAGE_ENDPOINT,
            api_key=AUTH_KEY,
        )
        adapter = VoyageAdapter(config)
        adapter.media_manager = mock_media_manager # 注入mock的media_manager
        return adapter

    def test_embedding(self, voyage_adapter: VoyageAdapter):
        req = LLMEmbeddingRequest(
            model="mock_embedding",
            inputs=[
                LLMChatTextContent(text="hello world", type="text"),
            ]
        )

        response = voyage_adapter.embed(req)
        assert isinstance(response, LLMEmbeddingResponse)
        assert response.vectors[0] == REFERENCE_VECTOR
        assert response.usage.total_tokens == 10

    def test_multi_modal_embedding(self, voyage_adapter: VoyageAdapter):
        req = LLMEmbeddingRequest(
            inputs=[
                LLMChatTextContent(text="hello world", type="text"),
                LLMChatImageContent(media_id="fd76f6fa-d7c7-4dfe-be48-bb2f7d87c9fb", type="image")
            ],
            model="mock_multimodal"
        )

        response = voyage_adapter.embed(req)
        assert isinstance(response, LLMEmbeddingResponse)
        assert response.vectors[0] == REFERENCE_VECTOR
        assert response.usage.total_tokens == 3576 #type: ignore

    def test_rerank_without_sort(self, voyage_adapter: VoyageAdapter):
        req = LLMReRankRequest(
            query="how are you?",
            documents=[
                "I'm doing well, thank you.",
                "I'm fine, thank you."                
            ],
            model="mock_rerank",
            return_documents=True
        )


        response = voyage_adapter.rerank(req)
        assert isinstance(response, LLMReRankResponse)
        assert response.contents[0].document == "I'm doing well, thank you."
        assert response.contents[1].document == "I'm fine, thank you."
        assert response.contents[0].score == 0.4375
        assert response.contents[1].score == 0.421875
        assert response.usage.total_tokens == 26

    def test_rerank_with_sort(self, voyage_adapter: VoyageAdapter):
        req = LLMReRankRequest(
            query="how are you?",
            documents=[
                "I'm doing well, thank you.",
                "I'm fine, thank you."                
            ],
            model="mock_rerank",
            return_documents=True,
            sort=True
        )

        response = voyage_adapter.rerank(req)
        assert isinstance(response, LLMReRankResponse)
        assert response.contents[0].score > response.contents[1].score

    def test_rerank_sort_raise_error(self, voyage_adapter: VoyageAdapter):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            req = LLMReRankRequest(
                query="how are you?",
                documents=[
                    "I'm doing well, thank you.",
                    "I'm fine, thank you."                
                ],
                model="mock_rerank",
                sort=True
            )