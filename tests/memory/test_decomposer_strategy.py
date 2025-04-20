from datetime import datetime
from unittest.mock import Mock

import pytest

from kirara_ai.im.sender import ChatSender, ChatType
from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.llm.format.message import (LLMChatImageContent, LLMChatMessage, LLMChatTextContent, LLMToolCallContent,
                                          LLMToolResultContent)
from kirara_ai.memory.composes.decomposer_strategy import (ContentInfo, ContentParser, DefaultDecomposerStrategy,
                                                           MediaContentStrategy, MultiElementDecomposerStrategy,
                                                           TextContentStrategy, ToolCallContentStrategy,
                                                           ToolResultContentStrategy)
from kirara_ai.memory.entry import MemoryEntry


@pytest.fixture
def mock_container():
    container = Mock(spec=DependencyContainer)
    media_manager = Mock()
    container.resolve.return_value = media_manager
    return container


@pytest.fixture
def sample_entry():
    entry = MemoryEntry(
        sender=ChatSender(user_id="user1", chat_type=ChatType.C2C, display_name="Test User"),
        content="这是一段文本 <media_msg id=\"media1\" /> 继续文本 <function_call id=\"call1\" name=\"test_function\" /> 更多文本 <tool_result id=\"result1\" name=\"test_result\" isError=\"false\" />",
        timestamp=datetime.now(),
        metadata={
            "_media_ids": ["media1"],
            "_tool_calls": [{
                "id": "call1",
                "name": "test_function",
                "arguments": {"arg1": "value1"}
            }],
            "_tool_results": [{
                "id": "result1",
                "name": "test_result",
                "isError": False,
                "content": [{"type": "text", "text": "结果文本"}]
            }]
        }
    )
    return entry


class TestTextContentStrategy:
    def test_extract_content(self, sample_entry):
        strategy = TextContentStrategy()
        content_infos = strategy.extract_content(sample_entry.content, sample_entry)

        assert len(content_infos) == 3
        assert content_infos[0].content_type == "text"
        assert content_infos[0].text == "这是一段文本"
        assert content_infos[1].text == "继续文本"
        assert content_infos[2].text == "更多文本"

    def test_to_llm_content(self):
        strategy = TextContentStrategy()
        info = ContentInfo(
            content_type="text",
            start=0,
            end=10,
            text="测试文本"
        )

        content = strategy.to_llm_content(info)
        assert isinstance(content, LLMChatTextContent)
        assert content.text == "测试文本"

    def test_to_text(self):
        strategy = TextContentStrategy()
        info = ContentInfo(
            content_type="text",
            start=0,
            end=10,
            text="测试文本"
        )

        text = strategy.to_text(info)
        assert text == "测试文本"


class TestMediaContentStrategy:
    def test_extract_content(self, sample_entry):
        strategy = MediaContentStrategy()
        content_infos = strategy.extract_content(sample_entry.content, sample_entry)

        assert len(content_infos) == 1
        assert content_infos[0].content_type == "media"
        assert content_infos[0].metadata["media_id"] == "media1"

    def test_to_llm_content(self):
        strategy = MediaContentStrategy()
        info = ContentInfo(
            content_type="media",
            start=0,
            end=10,
            text="<media_msg id=\"media1\" />",
            metadata={"media_id": "media1"}
        )

        content = strategy.to_llm_content(info)
        assert isinstance(content, LLMChatImageContent)
        assert content.media_id == "media1"

    def test_to_text(self):
        strategy = MediaContentStrategy()
        info = ContentInfo(
            content_type="media",
            start=0,
            end=10,
            text="<media_msg id=\"media1\" />",
            metadata={"media_id": "media1"}
        )

        text = strategy.to_text(info)
        assert text == "<media_msg id=\"media1\" />"


class TestToolCallContentStrategy:
    def test_extract_content(self, sample_entry):
        strategy = ToolCallContentStrategy()
        content_infos = strategy.extract_content(sample_entry.content, sample_entry)

        assert len(content_infos) == 1
        assert content_infos[0].content_type == "tool_call"
        assert content_infos[0].metadata["id"] == "call1"
        assert content_infos[0].metadata["name"] == "test_function"

    def test_no_metadata_returns_empty(self):
        strategy = ToolCallContentStrategy()
        entry = MemoryEntry(
            content="<function_call id=\"call1\" name=\"test_function\" />",
            sender=ChatSender(user_id="user1", chat_type=ChatType.C2C, display_name="Test User")
        )

        content_infos = strategy.extract_content(entry.content, entry)
        assert len(content_infos) == 0

    def test_to_llm_content(self):
        strategy = ToolCallContentStrategy()
        info = ContentInfo(
            content_type="tool_call",
            start=0,
            end=10,
            text="<function_call id=\"call1\" name=\"test_function\" />",
            metadata={
                "id": "call1",
                "name": "test_function",
                "arguments": {"arg1": "value1"}
            }
        )

        content = strategy.to_llm_content(info)
        assert isinstance(content, LLMToolCallContent)
        assert content.id == "call1"
        assert content.name == "test_function"

    def test_to_text(self):
        strategy = ToolCallContentStrategy()
        info = ContentInfo(
            content_type="tool_call",
            start=0,
            end=10,
            text="<function_call id=\"call1\" name=\"test_function\" />",
            metadata={
                "id": "call1",
                "name": "test_function"
            }
        )

        text = strategy.to_text(info)
        assert text == "<function_call id=\"call1\" name=\"test_function\" />"


class TestToolResultContentStrategy:
    def test_extract_content(self, sample_entry):
        strategy = ToolResultContentStrategy()
        content_infos = strategy.extract_content(sample_entry.content, sample_entry)

        assert len(content_infos) == 1
        assert content_infos[0].content_type == "tool_result"
        assert content_infos[0].metadata["id"] == "result1"
        assert content_infos[0].metadata["name"] == "test_result"
        assert content_infos[0].metadata["isError"] is False

    def test_to_llm_content(self):
        strategy = ToolResultContentStrategy()
        info = ContentInfo(
            content_type="tool_result",
            start=0,
            end=10,
            text="<tool_result id=\"result1\" name=\"test_result\" isError=\"false\" />",
            metadata={
                "id": "result1",
                "name": "test_result",
                "isError": False,
                "content": [{"type": "text", "text": "结果文本"}]
            }
        )

        content = strategy.to_llm_content(info)
        assert isinstance(content, LLMToolResultContent)
        assert content.id == "result1"
        assert content.name == "test_result"
        assert content.isError is False

    def test_to_text(self):
        strategy = ToolResultContentStrategy()
        info = ContentInfo(
            content_type="tool_result",
            start=0,
            end=10,
            text="<tool_result id=\"result1\" name=\"test_result\" isError=\"false\" />",
            metadata={
                "id": "result1",
                "name": "test_result",
                "isError": False
            }
        )

        text = strategy.to_text(info)
        assert text == "<tool_result id=\"result1\" name=\"test_result\" isError=\"False\" />"


class TestContentParser:
    def test_parse_content(self, sample_entry):
        parser = ContentParser()
        content_infos = parser.parse_content(sample_entry.content, sample_entry)

        assert len(content_infos) == 6  # 3 text parts + 1 media + 1 tool call + 1 tool result = 6

        # 检查内容是否按位置排序
        for i in range(len(content_infos) - 1):
            assert content_infos[i].start < content_infos[i + 1].start

    def test_to_llm_message(self):
        parser = ContentParser()
        content_infos = [
            ContentInfo(content_type="text", start=0, end=10, text="Hello"),
            ContentInfo(content_type="media", start=11, end=20, text="<media_msg id=\"media1\" />", metadata={"media_id": "media1"})
        ]

        message = parser.to_llm_message(content_infos, "user")[0]
        assert isinstance(message, LLMChatMessage)
        assert message.role == "user"
        assert len(message.content) == 2
        assert isinstance(message.content[0], LLMChatTextContent)
        assert isinstance(message.content[1], LLMChatImageContent)

    def test_to_text(self):
        parser = ContentParser()
        content_infos = [
            ContentInfo(content_type="text", start=0, end=10, text="Hello"),
            ContentInfo(content_type="media", start=11, end=20, text="<media_msg id=\"media1\" />", metadata={"media_id": "media1"})
        ]

        text = parser.to_text(content_infos)
        assert text == "Hello<media_msg id=\"media1\" />"


class TestDefaultDecomposerStrategy:
    def test_decompose_empty_entries(self):
        strategy = DefaultDecomposerStrategy()
        result = strategy.decompose([], {"empty_message": "空消息"})

        assert len(result) == 1
        assert result[0] == "空消息"

    def test_decompose_with_entries(self, sample_entry):
        strategy = DefaultDecomposerStrategy()
        result = strategy.decompose([sample_entry], {})

        assert len(result) == 1
        assert isinstance(result[0], str)
        assert "这是一段文本" in result[0]
        assert "<media_msg id=\"media1\" />" in result[0]


class TestMultiElementDecomposerStrategy:
    def test_decompose_empty_entries(self):
        strategy = MultiElementDecomposerStrategy()
        result = strategy.decompose([], {})

        assert len(result) == 0

    def test_process_entry_user_content(self):
        strategy = MultiElementDecomposerStrategy()
        entry = MemoryEntry(
            sender=ChatSender(user_id="user1", chat_type=ChatType.C2C, display_name="Test User"),
            content="用户消息",
            timestamp=datetime.now()
        )

        messages = strategy._process_entry(entry)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert len(messages[0].content) == 1
        assert isinstance(messages[0].content[0], LLMChatTextContent)
        assert messages[0].content[0].text == "用户消息"

    def test_process_entry_with_ai_response(self):
        strategy = MultiElementDecomposerStrategy()
        entry = MemoryEntry(
            sender=ChatSender(user_id="user1", chat_type=ChatType.C2C, display_name="Test User"),
            content="用户消息\n你回答: AI回复",
            timestamp=datetime.now()
        )

        messages = strategy._process_entry(entry)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert isinstance(messages[0].content[0], LLMChatTextContent)
        assert messages[0].content[0].text == "用户消息"
        assert messages[1].role == "assistant"
        assert isinstance(messages[1].content[0], LLMChatTextContent)
        assert messages[1].content[0].text == "AI回复"

    def test_merge_adjacent_messages(self):
        strategy = MultiElementDecomposerStrategy()
        messages = [
            LLMChatMessage(role="user", content=[LLMChatTextContent(text="消息1")]),
            LLMChatMessage(role="user", content=[LLMChatTextContent(text="消息2")]),
            LLMChatMessage(role="assistant", content=[LLMChatTextContent(text="回复")])
        ]

        strategy._merge_adjacent_messages(messages)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert len(messages[0].content) == 2
        assert messages[1].role == "assistant"
