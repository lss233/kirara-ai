from unittest.mock import Mock

import pytest

import kirara_ai.llm.format.tool as tools
from kirara_ai.im.message import ImageMessage, IMMessage, TextMessage
from kirara_ai.im.sender import ChatSender, ChatType
from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.llm.format.message import (LLMChatImageContent, LLMChatMessage, LLMChatTextContent, LLMToolCallContent,
                                          LLMToolResultContent)
from kirara_ai.memory.composes.composer_strategy import (IMMessageProcessor, LLMChatImageContentProcessor,
                                                         LLMChatMessageProcessor, LLMChatTextContentProcessor,
                                                         LLMToolCallContentProcessor, LLMToolResultContentProcessor,
                                                         MediaMessageProcessor, ProcessorFactory, TextMessageProcessor,
                                                         drop_think_part)


@pytest.fixture
def mock_container():
    container = Mock(spec=DependencyContainer)
    media_manager = Mock()
    container.resolve.return_value = media_manager
    return container


@pytest.fixture
def sample_context():
    return {
        "media_ids": [],
        "tool_calls": [],
        "tool_results": []
    }


class TestDropThinkPart:
    def test_drop_think_part_with_think_tag(self):
        text = "<think>这是思考部分</think>这是输出部分"
        result = drop_think_part(text)
        assert result == "这是输出部分"

    def test_drop_think_part_without_think_tag(self):
        text = "这是纯文本，没有思考标签"
        result = drop_think_part(text)
        assert result == text

class TestTextMessageProcessor:
    def test_process(self, mock_container, sample_context):
        processor = TextMessageProcessor(mock_container)
        message = TextMessage("这是一条文本消息")

        result = processor.process(message, sample_context)
        assert result == "这是一条文本消息\n"


class TestMediaMessageProcessor:
    def test_process(self, mock_container, sample_context):
        processor = MediaMessageProcessor(mock_container)
        message = ImageMessage(media_id="media1", data=b"test", format="png")

        result = processor.process(message, sample_context)
        assert "id=\"media1\"" in result
        assert sample_context["media_ids"] == ["media1"]


class TestLLMChatTextContentProcessor:
    def test_process_normal_text(self, mock_container, sample_context):
        processor = LLMChatTextContentProcessor(mock_container)
        content = LLMChatTextContent(text="这是普通文本")

        result = processor.process(content, sample_context)
        assert result == "这是普通文本\n"

    def test_process_with_think_tag(self, mock_container, sample_context):
        processor = LLMChatTextContentProcessor(mock_container)
        content = LLMChatTextContent(text="<think>思考过程</think>这是输出")

        result = processor.process(content, sample_context)
        assert result == "这是输出\n"


class TestLLMChatImageContentProcessor:
    def test_process(self, mock_container, sample_context):
        # 设置 media_manager mock
        media_manager = mock_container.resolve.return_value
        media = Mock()
        media.description = "图片描述"
        media_manager.get_media.return_value = media

        processor = LLMChatImageContentProcessor(mock_container)
        content = LLMChatImageContent(media_id="media1")

        result = processor.process(content, sample_context)
        assert "media_msg" in result
        assert "media1" in result
        assert "图片描述" in result
        assert sample_context["media_ids"] == ["media1"]


class TestLLMToolCallContentProcessor:
    def test_process(self, mock_container, sample_context):
        processor = LLMToolCallContentProcessor(mock_container)
        content = LLMToolCallContent(
            id="call1",
            name="test_function",
            parameters={"arg1": "value1"}
        )

        result = processor.process(content, sample_context)
        assert "function_call" in result
        assert "id=\"call1\"" in result
        assert "name=\"test_function\"" in result
        assert len(sample_context["tool_calls"]) == 1

        # 检查添加到上下文的工具调用数据
        tool_call = sample_context["tool_calls"][0]
        assert tool_call["id"] == "call1"
        assert tool_call["name"] == "test_function"


class TestLLMToolResultContentProcessor:
    def test_process(self, mock_container, sample_context):
        processor = LLMToolResultContentProcessor(mock_container)
        content = LLMToolResultContent(
            id="result1",
            name="test_result",
            isError=False,
            content=[tools.TextContent(text="结果文本")]
        )

        result = processor.process(content, sample_context)
        assert "tool_result" in result
        assert "id=\"result1\"" in result
        assert "name=\"test_result\"" in result
        assert "isError=\"False\"" in result
        assert len(sample_context["tool_results"]) == 1

        # 检查添加到上下文的工具结果数据
        tool_result = sample_context["tool_results"][0]
        assert tool_result["id"] == "result1"
        assert tool_result["name"] == "test_result"
        assert tool_result["isError"] is False


class TestIMMessageProcessor:
    def test_process_with_text_message(self, mock_container, sample_context):
        # 创建带有文本消息的 IMMessage
        text_message = TextMessage("这是文本消息")
        im_message = IMMessage(
            sender=ChatSender(user_id="user1", chat_type=ChatType.C2C, display_name="用户"),
            message_elements=[text_message]
        )

        processor = IMMessageProcessor(mock_container)
        result = processor.process(im_message, sample_context)

        assert "用户 说:" in result
        assert "这是文本消息" in result

    def test_process_with_media_message(self, mock_container, sample_context):
        # 创建带有媒体消息的 IMMessage
        media_message = ImageMessage(media_id="media1", data=b"test", format="png")
        im_message = IMMessage(
            sender=ChatSender(user_id="user1", chat_type=ChatType.C2C, display_name="用户"),
            message_elements=[media_message]
        )

        processor = IMMessageProcessor(mock_container)
        result = processor.process(im_message, sample_context)

        assert "用户 说:" in result
        assert "media_msg" in result
        assert "media1" in result
        assert sample_context["media_ids"] == ["media1"]


class TestLLMChatMessageProcessor:
    def test_process_with_text_content(self, mock_container, sample_context):
        message = LLMChatMessage(
            role="user",
            content=[LLMChatTextContent(text="这是文本内容")]
        )

        processor = LLMChatMessageProcessor(mock_container)
        result = processor.process(message, sample_context)

        assert "你回答:" in result
        assert "这是文本内容" in result

    def test_process_with_mixed_content(self, mock_container, sample_context):
        # 设置 media_manager mock
        media_manager = mock_container.resolve.return_value
        media = Mock()
        media.description = "图片描述"
        media_manager.get_media.return_value = media

        message = LLMChatMessage(
            role="assistant",
            content=[
                LLMChatTextContent(text="这是文本内容"),
                LLMChatImageContent(media_id="media1")
            ]
        )

        processor = LLMChatMessageProcessor(mock_container)
        result = processor.process(message, sample_context)

        assert "你回答:" in result
        assert "这是文本内容" in result
        assert "media_msg" in result
        assert "media1" in result
        assert sample_context["media_ids"] == ["media1"]

    def test_process_with_tool_content(self, mock_container, sample_context):
        message = LLMChatMessage(
            role="assistant",
            content=[LLMToolCallContent(
                id="call1",
                name="test_function",
                parameters={"arg1": "value1"}
            )]
        )

        processor = LLMChatMessageProcessor(mock_container)
        result = processor.process(message, sample_context)

        assert "function_call" in result
        assert "id=\"call1\"" in result
        assert "name=\"test_function\"" in result
        assert len(sample_context["tool_calls"]) == 1


class TestProcessorFactory:
    def test_get_processor_for_im_message(self, mock_container):
        factory = ProcessorFactory(mock_container)
        processor = factory.get_processor(IMMessage)

        assert isinstance(processor, IMMessageProcessor)

    def test_get_processor_for_llm_chat_message(self, mock_container):
        factory = ProcessorFactory(mock_container)
        processor = factory.get_processor(LLMChatMessage)

        assert isinstance(processor, LLMChatMessageProcessor)

    def test_get_processor_unknown_type(self, mock_container):
        factory = ProcessorFactory(mock_container)
        processor = factory.get_processor(object)

        assert processor is None
