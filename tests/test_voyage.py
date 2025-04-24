from kirara_ai.plugins.llm_preset_adapters.voyage_adapter import resolve_media_base64
from kirara_ai.llm.format.message import LLMChatTextContent, LLMChatImageContent

import pytest
import asyncio

class MockMediaManager:
    def get_media(self, media_id: str):
        return MockMedia(media_id)
    
class MockMedia:
    def __init__(self, media_id: str):
        self.media_id = media_id

    async def get_base64(self) -> str:
        return f"{self.media_id}:base64_data"
    
    def __repr__(self):
        return f"MockMedia(media_id={self.media_id}), expected_base64_data={self.media_id}:base64_data"

@pytest.fixture
def media_manager():
    return MockMediaManager()

@pytest.fixture
def inputs():
    return [
        LLMChatTextContent(type="text",text="hello"),
        LLMChatImageContent(type="image",media_id="114514"),
        LLMChatImageContent(type="image",media_id="1919810"),
        LLMChatTextContent(type="text",text="world"),
    ]

def test_resolve_media_base64(inputs, media_manager):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    results = loop.run_until_complete(resolve_media_base64(inputs, media_manager))
    assert results[0] == {'content': [{"type": "text", "text": "hello"}]}
    assert results[1] == {"content": [{"type": "image_base64", "image_base64": "114514:base64_data"}]}
    assert results[2] == {"content": [{"type": "image_base64", "image_base64": "1919810:base64_data"}]}
    assert results[3] == {'content': [{"type": "text", "text": "world"}]}