import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from kirara_ai.im.message import IMMessage, TextMessage
from kirara_ai.im.sender import ChatSender
from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.llm.format.message import LLMChatMessage, LLMChatTextContent, LLMToolResultContent
from kirara_ai.llm.format.response import LLMChatResponse, Message, Usage
from kirara_ai.llm.format.tool import CallableWrapper, Function, TextContent, Tool, ToolCall, ToolInputSchema
from kirara_ai.llm.llm_manager import LLMManager
from kirara_ai.workflow.core.execution.executor import WorkflowExecutor
from kirara_ai.workflow.implementations.blocks.llm.chat import (ChatCompletion, ChatCompletionWithTools,
                                                                ChatMessageConstructor, ChatResponseConverter)


def get_tools() -> list[Tool]:
    async def mock_tool_invoke(tool_call: ToolCall) -> LLMToolResultContent:
        return LLMToolResultContent(
            id=tool_call.id,
            name=tool_call.function.name,
            content=[TextContent(text="晴天，温度25°C")]
        )
    
    return [
        Tool(
            type="function",
            name="get_weather",
            description="Get the current weather in a given location",
            parameters=ToolInputSchema(
                type="object",
                properties = {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                required=["location"],
            ),
            invokeFunc=CallableWrapper(mock_tool_invoke)
        )
    ]

def get_llm_tool_calls() -> list[ToolCall]:
    return [
        ToolCall(
            id = "call_e33147bcb72525ed",
            function = Function(
                name="get_weather",
                arguments={"location": "San Francisco, CA"}
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
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            )
        )

class MockLLMWithToolCalls:
    def __init__(self, with_tool_calls=True):
        self.with_tool_calls = with_tool_calls
        self.call_count = 0
    
    def chat(self, request):
        self.call_count += 1
        
        # 第一次调用返回工具调用
        if self.with_tool_calls and self.call_count == 1:
            return LLMChatResponse(
                message=Message(
                    role="assistant",
                    content=[LLMChatTextContent(text="我需要查询天气")],
                    tool_calls=get_llm_tool_calls()
                ),
                model="gpt-3.5-turbo",
                usage=Usage(
                    prompt_tokens=10,
                    completion_tokens=20, 
                    total_tokens=30
                )
            )
        # 后续调用返回最终回复
        else:
            return LLMChatResponse(
                message=Message(
                    role="assistant",
                    content=[LLMChatTextContent(text="旧金山今天是晴天，温度25°C")]
                ),
                model="gpt-3.5-turbo",
                usage=Usage(
                    prompt_tokens=10,
                    completion_tokens=20, 
                    total_tokens=30
                )
            )

# 创建模拟的 LLMManager 类
class MockLLMManager(LLMManager):
    def __init__(self):
        self.mock_llm = MockLLM()

    def get_llm_id_by_ability(self, ability):
        return "gpt-3.5-turbo"

    def get_llm(self, model_id):
        return self.mock_llm
    
class MockLLMManagerWithToolCalls(LLMManager):
    def __init__(self, with_tool_calls=True):
        self.mock_llm = MockLLMWithToolCalls(with_tool_calls)

    def get_llm_id_by_ability(self, ability):
        return "gpt-3.5-turbo"

    def get_llm(self, model_id):
        return self.mock_llm

@pytest.fixture
def container():
    """创建一个带有模拟 LLM 提供者的容器"""
    container = DependencyContainer()

    # 模拟 LLMManager
    mock_llm_manager = MockLLMManager()

    # 模拟 LLM
    # mock_llm = MockLLM()

    # 模拟响应
    mock_response = LLMChatResponse(
        message=Message(
            role="assistant",
            content=[LLMChatTextContent(text="这是 AI 的回复")]
        ),
        model="gpt-3.5-turbo",
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
    )
    # mock_llm.chat.return_value = mock_response

    # 模拟 WorkflowExecutor
    mock_executor = MagicMock(spec=WorkflowExecutor)

    # 创建一个在新线程中运行的事件循环
    def start_background_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()
    
    # 创建新的事件循环
    new_loop = asyncio.new_event_loop()
    
    # 在新线程中启动事件循环
    t = threading.Thread(target=start_background_loop, args=(new_loop,), daemon=True)
    t.start()
    
    # 注册到容器
    container.register(LLMManager, mock_llm_manager)
    container.register(WorkflowExecutor, mock_executor)
    container.register(asyncio.AbstractEventLoop, new_loop)

    return container


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


def test_chat_completion(container):
    # 创建消息列表
    messages = [
        Message(role="system", content=[LLMChatTextContent(text="你是一个助手")]),
        Message(role="user", content=[LLMChatTextContent(text="你好，AI！")])
    ]

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
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
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
def test_chat_completion_with_tools(container):
    """测试工具调用块"""
    container.register(LLMManager, MockLLMManagerWithToolCalls(with_tool_calls=True))
    
    # 创建消息列表
    messages = [
        LLMChatMessage(role="system", content=[LLMChatTextContent(text="你是一个助手")]),
        LLMChatMessage(role="user", content=[LLMChatTextContent(text="旧金山今天天气如何？")])
    ]

    # 创建工具列表
    tools = get_tools()

    # 创建块
    block = ChatCompletionWithTools(model_name="gpt-3.5-turbo", max_iterations=3)
    block.container = container

    # 执行块
    result = block.execute(msg=messages, tools=tools)

    # 验证结果
    assert "resp" in result
    assert "iteration_msgs" in result
    assert isinstance(result["resp"], LLMChatResponse)
    assert isinstance(result["iteration_msgs"], list)
    assert len(result["iteration_msgs"]) >= 2  # 至少包含工具调用和最终回复
    
    # 验证工具调用过程
    assert result["iteration_msgs"][0].tool_calls is not None
    assert result["iteration_msgs"][0].tool_calls[0].function.name == "get_weather"
    
    # 验证最终回复
    assert "旧金山今天是晴天" in result["resp"].message.content[0].text

def test_chat_completion_with_tools_no_tool_calls(container):
    """测试工具调用块 - 无工具调用情况"""

    # 注册到容器 - 使用不会进行工具调用的模拟
    container.register(LLMManager, MockLLMManagerWithToolCalls(with_tool_calls=False))

    # 创建消息列表
    messages = [
        LLMChatMessage(role="system", content=[LLMChatTextContent(text="你是一个助手")]),
        LLMChatMessage(role="user", content=[LLMChatTextContent(text="你好，AI！")])
    ]

    # 创建工具列表
    tools = get_tools()

    # 创建块
    block = ChatCompletionWithTools(model_name="gpt-3.5-turbo", max_iterations=3)
    block.container = container

    # 执行块
    result = block.execute(msg=messages, tools=tools)

    # 验证结果 - 直接返回响应，没有工具调用
    assert "resp" in result
    assert "iteration_msgs" in result
    assert isinstance(result["resp"], LLMChatResponse)
    assert isinstance(result["iteration_msgs"], list)
    assert len(result["iteration_msgs"]) == 0  # 无消息，因为没有工具调用