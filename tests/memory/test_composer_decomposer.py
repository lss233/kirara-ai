from datetime import datetime
from unittest.mock import MagicMock

import pytest

import kirara_ai.llm.format.tool as tools
from kirara_ai.im.message import IMMessage, TextMessage
from kirara_ai.im.sender import ChatSender
from kirara_ai.llm.format.message import LLMChatMessage, LLMChatTextContent, LLMToolCallContent, LLMToolResultContent
from kirara_ai.memory.composes import DefaultMemoryComposer, DefaultMemoryDecomposer, MultiElementDecomposer


@pytest.fixture
def composer():
    return DefaultMemoryComposer()


@pytest.fixture
def decomposer():
    return DefaultMemoryDecomposer()


@pytest.fixture
def multi_decomposer():
    return MultiElementDecomposer()

@pytest.fixture
def group_sender():
    return ChatSender.from_group_chat(
        user_id="user1", group_id="group1", display_name="user1"
    )


@pytest.fixture
def c2c_sender():
    return ChatSender.from_c2c_chat(user_id="user1", display_name="user1")


class TestDefaultMemoryComposer:
    def test_compose_group_message(self, composer, group_sender):
        message = IMMessage(
            sender=group_sender,
            message_elements=[TextMessage(text="test message")],
        )

        entry = composer.compose(group_sender, [message])

        assert f"{group_sender.display_name} 说: \n{message.content}" in entry.content
        assert isinstance(entry.timestamp, datetime)

    def test_compose_c2c_message(self, composer, c2c_sender):
        message = IMMessage(
            sender=c2c_sender,
            message_elements=[TextMessage(text="test message")],
        )

        entry = composer.compose(c2c_sender, [message])

        assert f"{c2c_sender.display_name} 说: \n{message.content}" in entry.content
        assert isinstance(entry.timestamp, datetime)

    def test_compose_llm_response(self, composer, c2c_sender):
        chat_message = LLMChatMessage(role="assistant", content=[LLMChatTextContent(text="test response")])

        entry = composer.compose(c2c_sender, [chat_message])
        
        assert isinstance(chat_message.content[0], LLMChatTextContent)
        assert f"你回答: \n{chat_message.content[0].text}" in entry.content
        assert isinstance(entry.timestamp, datetime)
    
    def test_compose_llm_tool_call_message(self, composer, c2c_sender):
        chat_message = LLMChatMessage(role="assistant", content=[LLMChatTextContent(text="<think>我决定调用get_weather函数并传递city=北京。</think>"), LLMToolCallContent(name="get_weather", parameters={"city": "北京"})])

        entry = composer.compose(c2c_sender, [chat_message])

        # 判断其是否为空字符串
        assert entry.content == ""
        # 是否metadata中 _tool_calls 字段为非空列表
        assert entry.metadata.get("_tool_calls", None) is not None

    def test_compose_llm_tool_result_message(self, composer, c2c_sender):
        chat_message = LLMChatMessage(role = "tool", content = [LLMToolResultContent(name = "get_weather", content = [tools.TextContent(text="今天的天气是晴天。")])])

        entry = composer.compose(c2c_sender, [chat_message])

        assert entry.content == ""
        assert entry.metadata.get("_tool_results", None) is not None

class TestDefaultMemoryDecomposer:
    def test_decompose_mixed_entries(self, decomposer, group_sender, c2c_sender):
        entries = [
            MagicMock(
                sender=group_sender,
                content="group1:user1 说: group message",
                timestamp=datetime.now(),
            ),
            MagicMock(
                sender=c2c_sender,
                content="c2c:user1 说: c2c message",
                timestamp=datetime.now(),
            ),
        ]

        result = decomposer.decompose(entries)

        assert len(result) == 2
        assert "刚刚" in result[0]
        assert "group message" in result[0]
        assert "c2c message" in result[1]

    def test_decompose_empty(self, decomposer):
        result = decomposer.decompose([])
        assert result == [decomposer.empty_message]

    def test_decompose_max_entries(self, decomposer, c2c_sender):
        # 创建超过10条的记录
        entries = [
            MagicMock(
                sender=c2c_sender, content=f"message {i}", timestamp=datetime.now()
            )
            for i in range(12)
        ]

        result = decomposer.decompose(entries)

        # 验证只返回最后10条
        assert len(result) == 10
        assert "message 11" in result[-1]

class TestMultiElementDecomposer:
    def test_decompose_tool_call_and_result_message(self, multi_decomposer, c2c_sender):
        entries = [
            MagicMock(
                sender=c2c_sender,
                content="",
                timestamp=datetime.now(),
                metadata={"_tool_calls": [LLMToolCallContent(id ="call_114514",name="get_weather", parameters={"city": "北京"}).model_dump_json()]},
            ),
            MagicMock(
                sender=c2c_sender,
                content="",
                timestamp=datetime.now(),
                metadata={"_tool_results": [LLMToolResultContent(id="call_114514", name="get_weather", content=[tools.TextContent(text="今天的天气是晴天。")]).model_dump_json()]},
            )
        ]

        result = multi_decomposer.decompose(entries)

        tool_call_message = result[0]
        tool_result_message = result[1]

        assert tool_call_message.role == "assistant"
        assert all(isinstance(call, LLMToolCallContent) for call in tool_call_message.content)
        assert tool_result_message.role == "tool"
        assert all(isinstance(result, LLMToolResultContent) for result in tool_result_message.content)