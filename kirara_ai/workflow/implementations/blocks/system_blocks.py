from kirara_ai.workflow.core.block.registry import BlockRegistry
from kirara_ai.workflow.implementations.blocks.im.basic import ExtractChatSender
from kirara_ai.workflow.implementations.blocks.llm.basic import LLMResponseToText
from kirara_ai.workflow.implementations.blocks.llm.image import SimpleStableDiffusionWebUI
from kirara_ai.workflow.implementations.blocks.mcp.tool import MCPToolProvider
from kirara_ai.workflow.implementations.blocks.memory.clear_memory import ClearMemory
from kirara_ai.workflow.implementations.blocks.system.basic import (CurrentTimeBlock, TextBlock, TextConcatBlock,
                                                                    TextExtractByRegexBlock, TextReplaceBlock)

from .game.dice import DiceRoll
from .game.gacha import GachaSimulator
from .im.messages import AppendIMMessage, GetIMMessage, IMMessageToText, SendIMMessage, TextToIMMessage
from .im.states import ToggleEditState
from .llm.chat import ChatCompletion, ChatCompletionWithTools, ChatMessageConstructor, ChatResponseConverter
from .memory.chat_memory import ChatMemoryQuery, ChatMemoryStore
from .system.help import GenerateHelp


def register_system_blocks(registry: BlockRegistry):
    """注册系统自带的 block"""
    # 基础 blocks
    registry.register("text_block", "internal", TextBlock, "基础：文本")
    registry.register("text_concat_block", "internal", TextConcatBlock, "基础：拼接文本")
    registry.register("text_replace_block", "internal", TextReplaceBlock, "基础：替换文本")
    registry.register("text_extract_by_regex_block", "internal", TextExtractByRegexBlock, "基础：正则表达式提取文本")
    registry.register("current_time_block", "internal", CurrentTimeBlock, "基础：当前时间")

    # IM 相关 blocks
    registry.register("get_message", "internal", GetIMMessage, "IM: 获取最新消息")
    registry.register("send_message", "internal", SendIMMessage, "IM: 发送消息")
    registry.register(
        "toggle_edit_state", "internal", ToggleEditState, "IM: 切换编辑状态"
    )
    registry.register(
        "extract_chat_sender", "internal", ExtractChatSender, "IM: 提取消息发送者"
    )
    registry.register("append_im_message", "internal", AppendIMMessage, "IM: 补充消息")
    registry.register("im_message_to_text", "internal", IMMessageToText, "IM: 消息转文本")
    registry.register("text_to_im_message", "internal", TextToIMMessage, "文本: 文本转消息")

    # LLM 相关 blocks
    registry.register("chat_memory_query", "internal", ChatMemoryQuery, "LLM: 查询记忆")
    registry.register(
        "chat_message_constructor",
        "internal",
        ChatMessageConstructor,
        "LLM: 构造对话记录",
    )
    registry.register("chat_completion", "internal", ChatCompletion, "LLM: 执行对话")
    registry.register("chat_completion_with_tools", "internal", ChatCompletionWithTools, "LLM: 执行对话并调用工具")
    registry.register(
        "chat_response_converter",
        "internal",
        ChatResponseConverter,
        "LLM->IM: 转换消息",
    )
    registry.register("chat_memory_store", "internal", ChatMemoryStore, "LLM: 存储记忆")
    registry.register("llm_response_to_text", "internal", LLMResponseToText, "LLM: 响应转文本")

    # 画图相关 blocks
    registry.register(
        "simple_stable_diffusion_webui",
        "internal",
        SimpleStableDiffusionWebUI,
        "画图: 简单 Stable Diffusion WebUI",
    )

    # 游戏相关 blocks
    registry.register("dice_roll", "game", DiceRoll, "游戏: 掷骰子")
    registry.register("gacha_simulator", "game", GachaSimulator, "游戏: 抽卡模拟")

    # 系统相关 blocks
    registry.register("generate_help", "system", GenerateHelp, "系统: 生成帮助")
    registry.register("clear_memory", "system", ClearMemory, "系统: 清空记忆")
    
    # MCP 相关 blocks
    registry.register("mcp_tool_provider", "mcp", MCPToolProvider, "MCP: 提供工具")
