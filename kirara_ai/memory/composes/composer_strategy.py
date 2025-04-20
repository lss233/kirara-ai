from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

import kirara_ai.llm.format.tool as tools
from kirara_ai.im.message import IMMessage, MediaMessage, TextMessage
from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.llm.format.message import (LLMChatImageContent, LLMChatMessage, LLMChatTextContent, LLMToolCallContent,
                                          LLMToolResultContent)
from kirara_ai.media.manager import MediaManager

from .xml_helper import XMLHelper


def drop_think_part(text: str) -> str:
    """移除思考部分的文本"""
    import re
    return re.sub(r"(?:<think>[\s\S]*?</think>)?([\s\S]*)", r"\1", text, flags=re.DOTALL)


class MessageProcessor(ABC):
    """消息处理策略的基类"""

    def __init__(self, container: DependencyContainer):
        self.container = container

    @abstractmethod
    def process(self, message: Any, context: Dict) -> str:
        """处理特定类型的消息，返回组合后的文本"""


class TextMessageProcessor(MessageProcessor):
    """处理文本消息的策略"""

    def process(self, message: TextMessage, context: Dict) -> str:
        return f"{message.to_plain()}\n"


class MediaMessageProcessor(MessageProcessor):
    """处理媒体消息的策略"""

    def process(self, message: MediaMessage, context: Dict) -> str:
        media_ids = context.setdefault("media_ids", [])
        media_ids.append(message.media_id)

        desc = message.get_description()
        tag = XMLHelper.create_xml_tag("media_msg", {"id": message.media_id, "desc": desc})
        return f"{tag}\n"


class LLMChatTextContentProcessor(MessageProcessor):
    """处理LLM文本内容的策略"""

    def process(self, content: LLMChatTextContent, context: Dict) -> str:
        return f"{drop_think_part(content.text)}\n"


class LLMChatImageContentProcessor(MessageProcessor):
    """处理LLM图像内容的策略"""

    def process(self, content: LLMChatImageContent, context: Dict) -> str:
        media_ids = context.setdefault("media_ids", [])
        media_ids.append(content.media_id)

        media_manager = self.container.resolve(MediaManager)
        media = media_manager.get_media(content.media_id)
        desc = (media.description or "") if media else ""

        tag = XMLHelper.create_xml_tag("media_msg", {"id": content.media_id, "desc": desc})
        return f"{tag}\n"


class LLMToolCallContentProcessor(MessageProcessor):
    """处理LLM工具调用内容的策略"""

    def process(self, content: LLMToolCallContent, context: Dict) -> str:
        tool_calls = context.setdefault("tool_calls", [])
        tool_calls.append(content.model_dump())
        # parameters 比较长，保存到 metadata 里。
        tag = XMLHelper.create_xml_tag("function_call", {
            "id": content.id,
            "name": content.name
        })
        return f"{tag}\n"


class LLMToolResultContentProcessor(MessageProcessor):
    """处理LLM工具结果内容的策略"""

    def process(self, content: LLMToolResultContent, context: Dict) -> str:
        tool_results = context.setdefault("tool_results", [])

        tool_content = []
        for item in content.content:
            if isinstance(item, tools.TextContent):
                tool_content.append({
                    "type": "text",
                    "text": item.text
                })
            elif isinstance(item, tools.MediaContent):
                # 注册 media_id 引用
                media_ids = context.setdefault("media_ids", [])
                media_ids.append(item.media_id)
                tool_content.append({
                    "type": "media",
                    "media_id": item.media_id
                })
        # content 比较长，保存到 metadata 里。
        tool_results.append({
            "id": content.id,
            "name": content.name,
            "isError": content.isError,
            "content": tool_content
        })
        tag = XMLHelper.create_xml_tag("tool_result", {
            "id": content.id,
            "name": content.name,
            "isError": str(content.isError)
        })
        return f"{tag}\n"


class IMMessageProcessor(MessageProcessor):
    """处理IM消息的策略"""

    def __init__(self, container: DependencyContainer):
        super().__init__(container)
        self.element_processors: Dict[Type, MessageProcessor] = {
            TextMessage: TextMessageProcessor(container),
            MediaMessage: MediaMessageProcessor(container)
        }

    def process(self, message: IMMessage, context: Dict) -> str:
        result = f"{message.sender.display_name} 说: \n"

        for element in message.message_elements:
            for process_type, processor in self.element_processors.items():
                if isinstance(element, process_type):
                    result += processor.process(element, context)
                    break
                else:
                    result += f"{element.to_plain()}\n"

        return result


class LLMChatMessageProcessor(MessageProcessor):
    """处理LLM聊天消息的策略"""

    def __init__(self, container: DependencyContainer):
        super().__init__(container)
        self.content_processors: Dict[Type, MessageProcessor] = {
            LLMChatTextContent: LLMChatTextContentProcessor(container),
            LLMChatImageContent: LLMChatImageContentProcessor(container),
            LLMToolCallContent: LLMToolCallContentProcessor(container),
            LLMToolResultContent: LLMToolResultContentProcessor(container)
        }

    def process(self, message: LLMChatMessage, context: Dict) -> str:
        result = ""
        temp = ""

        for part in message.content:
            part_type = type(part)
            for processor_type, processor in self.content_processors.items():
                if issubclass(part_type, processor_type):
                    if part_type in [LLMToolCallContent, LLMToolResultContent]:
                            # 工具调用和结果直接添加到结果中，不经过temp
                        result += processor.process(part, context)
                    else:
                        # 其他内容添加到temp中
                        temp += processor.process(part, context)

        if temp.strip("\n"):
            result += f"你回答: \n{temp}"

        return result


class ProcessorFactory:
    """消息处理器工厂，用于创建和管理不同类型消息的处理器"""

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.processors: Dict[Type, MessageProcessor] = {
            IMMessage: IMMessageProcessor(container),
            LLMChatMessage: LLMChatMessageProcessor(container)
        }

    def get_processor(self, message_type: Type) -> Optional[MessageProcessor]:
        """获取特定类型消息的处理器"""
        for processor_type, processor in self.processors.items():
            if issubclass(message_type, processor_type):
                return processor
        return None
