import asyncio
from unittest.mock import MagicMock, patch

import pytest

from kirara_ai.im.message import IMMessage, TextMessage
from kirara_ai.im.sender import ChatSender
from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.llm.format.message import LLMChatTextContent, LLMChatMessage
from kirara_ai.llm.format.response import LLMChatResponse, Message, ToolCall, Function
from kirara_ai.llm.format.request import Tool, ToolParameters, LLMChatRequest
from kirara_ai.llm.llm_manager import LLMManager
from kirara_ai.workflow.core.execution.executor import WorkflowExecutor
from kirara_ai.workflow.implementations.blocks.llm.chat import (ChatCompletion, ChatMessageConstructor,
                                                                ChatResponseConverter, FunctionCalling)


def get_tools() -> list[Tool]:
    return [
        Tool(
            type="function",
            name="get_weather",
            description="Get the current weather in a given location",
            parameters=ToolParameters(
                type="object",
                properties = {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                required=["location"],
            )
        )
    ]

def get_llm_tool_calls() -> list[ToolCall]:
    return [
        ToolCall(
            id = "call_e33147bcb72525ed",
            model = "openai",
            function = Function(
                name="get_weather",
                parameters={"location": "San Francisco, CA"}
            )
        )
    ]

# 创建模拟的 LLM 类
class MockLLM:
    def chat(self, request):
        return LLMChatResponse(
            message=Message(
                role="assistant",
                content=[LLMChatTextContent(text="这是 AI 的回复")]
            ),
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10,
                   "completion_tokens": 20, "total_tokens": 30}
        )

class MockLLMFunctionCalling:
    def chat(self, request):
        return LLMChatResponse(
            message=Message(
                role="assistant",
                content=[LLMChatTextContent(text="这是 AI 的回复")],
                tool_calls=get_llm_tool_calls()
            ),
            model="gpt-3.5-turbo",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20, 
                "total_tokens": 30
            }
        )

# 创建模拟的 LLMManager 类
class MockLLMManager(LLMManager):
    def __init__(self):
        self.mock_llm = MockLLM()

    def get_llm_id_by_ability(self, ability):
        return "gpt-3.5-turbo"

    def get_llm(self, model_id):
        return self.mock_llm
    
class MockLLMManagerFunctionCalling(MockLLMManager):
    def __init__(self):
        self.mock_llm = MockLLMFunctionCalling()

@pytest.fixture
def container():
    """创建一个带有模拟 LLM 提供者的容器"""
    container = DependencyContainer()

    # 模拟 LLMManager
    mock_llm_manager = MagicMock(spec=LLMManager)
    mock_llm_manager.get_llm_id_by_ability.return_value = "gpt-3.5-turbo"

    # 模拟 LLM
    mock_llm = MagicMock()
    mock_llm_manager.get_llm.return_value = mock_llm

    # 模拟响应
    mock_response = LLMChatResponse(
        message=Message(
            role="assistant",
            content=[LLMChatTextContent(text="这是 AI 的回复")]
        ),
        model="gpt-3.5-turbo",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    )
    mock_llm.chat.return_value = mock_response

    # 模拟 WorkflowExecutor
    mock_executor = MagicMock(spec=WorkflowExecutor)

    # 模拟 event loop
    mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    # 使用同步函数模拟 create_task，避免协程未等待的警告
    mock_loop.create_task = MagicMock()

    # 注册到容器
    container.register(LLMManager, mock_llm_manager)
    container.register(WorkflowExecutor, mock_executor)
    container.register(asyncio.AbstractEventLoop, mock_loop)

    return container, mock_llm_manager, mock_loop, mock_response


@patch('kirara_ai.workflow.implementations.blocks.llm.chat.ChatMessageConstructor.execute')
def test_chat_message_constructor(mock_execute):
    """测试聊天消息构造器"""
    # 模拟 execute 方法的返回值
    mock_execute.return_value = {
        "llm_msg": [Message(role="user", content=[LLMChatTextContent(text="你好，AI！")])]
    }

    # 创建块
    block = ChatMessageConstructor()

    # 模拟容器
    mock_container = MagicMock(spec=DependencyContainer)
    block.container = mock_container

    # 执行块 - 基本用法
    user_msg = IMMessage(
        sender=ChatSender.from_c2c_chat(
            user_id="test_user", display_name="Test User"),
        message_elements=[TextMessage("你好，AI！")]
    )

    result = block.execute(
        user_msg=user_msg,
        memory_content="",
        system_prompt_format="",
        user_prompt_format=""
    )

    # 验证结果
    assert "llm_msg" in result
    assert isinstance(result["llm_msg"], list)
    assert len(result["llm_msg"]) > 0
    assert result["llm_msg"][0].role == "user"
    assert result["llm_msg"][0].content[0].text == "你好，AI！"


@pytest.mark.asyncio
async def test_chat_completion_async():
    """使用 pytest-asyncio 测试聊天完成块"""
    # 创建容器
    container = DependencyContainer()

    # 创建消息列表
    messages = [
        Message(role="system", content=[LLMChatTextContent(text="你是一个助手")]),
        Message(role="user", content=[LLMChatTextContent(text="你好，AI！")])
    ]

    # 获取事件循环
    loop = asyncio.get_event_loop()

    # 注册到容器
    container.register(LLMManager, MockLLMManager())
    container.register(asyncio.AbstractEventLoop, loop)

    # 创建块 - 默认参数
    block = ChatCompletion()
    block.container = container

    # 执行块
    result = block.execute(prompt=messages)

    # 验证结果
    assert "resp" in result
    assert isinstance(result["resp"], LLMChatResponse)
    assert result["resp"].message.content[0].text == "这是 AI 的回复"


def test_chat_response_converter():
    """测试聊天响应转换器"""
    # 创建聊天响应
    chat_response = LLMChatResponse(
        message=Message(
            role="assistant",
            content=[LLMChatTextContent(text="这是 AI 的回复")]
        ),
        model="gpt-3.5-turbo",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    )

    # 创建块
    block = ChatResponseConverter()

    # 模拟容器
    mock_container = MagicMock(spec=DependencyContainer)
    # 模拟 get_bot_sender 方法
    mock_bot_sender = ChatSender.from_c2c_chat(
        user_id="bot", display_name="Bot")
    mock_container.resolve = MagicMock(
        side_effect=lambda x: mock_bot_sender if x == ChatSender.get_bot_sender else None)
    block.container = mock_container

    # 执行块
    result = block.execute(resp=chat_response)

    # 验证结果
    assert "msg" in result
    assert isinstance(result["msg"], IMMessage)
    assert "这是 AI 的回复" in result["msg"].content

@pytest.mark.asyncio
async def test_chat_function_calling():
    """测试函数调用块"""
    # 构造块需求参数
    chat_request = LLMChatRequest(
        model="gpt-3.5-turbo",
        tools=get_tools(),
        messages=[LLMChatMessage(role = "user", content=[LLMChatTextContent(text="今天天气如何？")])]
    )

    # 创建容器
    container = DependencyContainer()

    # 获取事件循环
    loop = asyncio.get_event_loop()

    # 注册到容器
    container.register(LLMManager, MockLLMManagerFunctionCalling())
    container.register(asyncio.AbstractEventLoop, loop)

    # 创建块
    block = FunctionCalling(model_name="gpt-3.5-turbo")
    block.container = container

    # 执行块 - step 1
    result = block.execute(request_body=chat_request)

    # 验证结果, llm 进行了函数调用请求
    assert "tool_call" in result
    assert "resp" not in result
    assert isinstance(result["tool_call"], LLMChatResponse)
    assert result["tool_call"].message.content[0].text == "这是 AI 的回复"
    assert result["tool_call"].message.tool_calls == get_llm_tool_calls()

    # 执行块 - step 2
    # 重新注册 LLMManager, 更换有 function calling 的llm
    container.register(LLMManager, MockLLMManager())
    result = block.execute(request_body=chat_request)

    # 验证结果, llm 未进行函数调用请求
    assert "resp" in result
    assert "tool_call" not in result
    assert isinstance(result["resp"], LLMChatResponse)
    assert result["resp"].message.content[0].text == "这是 AI 的回复"
    assert result["resp"].message.tool_calls is None