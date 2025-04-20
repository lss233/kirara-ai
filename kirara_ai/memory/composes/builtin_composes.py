from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from kirara_ai.im.message import IMMessage
from kirara_ai.im.sender import ChatSender
from kirara_ai.llm.format.message import LLMChatMessage
from kirara_ai.llm.format.response import Message
from kirara_ai.logger import get_logger
from kirara_ai.memory.entry import MemoryEntry

from .base import ComposableMessageType, MemoryComposer, MemoryDecomposer
from .composer_strategy import ProcessorFactory
from .decomposer_strategy import DefaultDecomposerStrategy, MultiElementDecomposerStrategy


class DefaultMemoryComposer(MemoryComposer):
    def __init__(self):
        self.processor_factory = None
    
    def compose(
        self, sender: Optional[ChatSender], message: List[ComposableMessageType]
    ) -> MemoryEntry:
        # 延迟初始化，确保 container 已被设置
        if self.processor_factory is None:
            self.processor_factory = ProcessorFactory(self.container)
        
        composed_message = ""
        # 上下文用于在处理过程中传递和收集数据
        context: Dict[str, Any] = {
            "media_ids": [],
            "tool_calls": [],
            "tool_results": []
        }
        
        for msg in message:
            msg_type = type(msg)
            processor = self.processor_factory.get_processor(msg_type)
            
            if processor:
                composed_message += processor.process(msg, context)
            elif isinstance(msg, str):
                # 处理字符串消息
                composed_message += f"{msg}\n"

        composed_message = composed_message.strip()
        composed_at = datetime.now()
        return MemoryEntry(
            sender=sender or ChatSender.get_bot_sender(),
            content=composed_message,
            timestamp=composed_at,
            metadata={
                "_media_ids": context.get("media_ids", []),
                "_tool_calls": context.get("tool_calls", []),
                "_tool_results": context.get("tool_results", []),
            },
        )


class DefaultMemoryDecomposer(MemoryDecomposer):
    def __init__(self):
        self.strategy = None
    
    def decompose(self, entries: List[MemoryEntry]) -> List[ComposableMessageType]:
        # 延迟初始化，确保 container 已被设置
        if self.strategy is None:
            self.strategy = DefaultDecomposerStrategy()
        
        # 使用上下文传递参数
        context = {
            "empty_message": self.empty_message
        }
        
        # 使用策略解析记忆条目
        return self.strategy.decompose(entries, context)


class MultiElementDecomposer(MemoryDecomposer):
    logger = get_logger("MultiElementDecomposer")
    
    def __init__(self):
        self.strategy = None
    
    def decompose(self, entries: List[MemoryEntry]) -> List[Union[IMMessage, LLMChatMessage, Message, str]]:
        # 延迟初始化，确保 container 已被设置
        if self.strategy is None:
            self.strategy = MultiElementDecomposerStrategy()
        
        # 使用上下文传递参数
        context = {
            "logger": self.logger
        }
        
        # 使用策略解析记忆条目
        return self.strategy.decompose(entries, context)
