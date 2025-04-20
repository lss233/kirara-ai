from datetime import datetime, timedelta
from typing import Any, Dict, List, NamedTuple, Protocol, cast

from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.llm.format.message import (LLMChatContentPartType, LLMChatImageContent, LLMChatMessage,
                                          LLMChatTextContent, LLMToolCallContent, LLMToolResultContent, RoleType)
from kirara_ai.logger import get_logger
from kirara_ai.memory.entry import MemoryEntry

from .base import ComposableMessageType
from .xml_helper import XMLHelper

logger = get_logger("DecomposerStrategy")


class ContentInfo(NamedTuple):
    """解析后的内容信息"""
    content_type: str  # 内容类型：text, media, tool_call, tool_result
    start: int  # 开始位置
    end: int  # 结束位置
    text: str  # 原始文本
    metadata: Dict[str, Any] = {}  # 相关元数据


class ContentParseStrategy(Protocol):
    """内容解析策略协议"""
    
    def extract_content(self, content: str, entry: MemoryEntry) -> List[ContentInfo]:
        """提取内容信息"""
        ...
    
    def to_llm_content(self, info: ContentInfo) -> LLMChatContentPartType:
        """转换为LLM内容类型"""
        ...
    
    def to_text(self, info: ContentInfo) -> str:
        """转换为文本形式"""
        ...


class TextContentStrategy:
    """文本内容解析策略"""
    
    def extract_content(self, content: str, entry: MemoryEntry) -> List[ContentInfo]:
        # 提取非标签文本
        text_parts = []
        current_pos = 0
        
        # 查找所有标签的位置
        tag_positions = []
        for tag_name in ["media_msg", "function_call", "tool_result"]:
            for _, start, end in XMLHelper.parse_xml_tag(content, tag_name):
                tag_positions.append((start, end))
        
        # 按位置排序
        tag_positions.sort()
        
        # 提取标签之间的文本
        for start, end in tag_positions:
            if start > current_pos:
                text = content[current_pos:start].strip()
                if text:
                    text_parts.append(ContentInfo(
                        content_type="text",
                        start=current_pos,
                        end=start,
                        text=text
                    ))
            current_pos = end
        
        # 处理最后一段文本
        if current_pos < len(content):
            text = content[current_pos:].strip()
            if text:
                text_parts.append(ContentInfo(
                    content_type="text",
                    start=current_pos,
                    end=len(content),
                    text=text
                ))
        
        return text_parts
    
    def to_llm_content(self, info: ContentInfo) -> LLMChatContentPartType:
        return LLMChatTextContent(text=info.text)
    
    def to_text(self, info: ContentInfo) -> str:
        return info.text


class MediaContentStrategy:
    """媒体内容解析策略"""
    
    def __init__(self, container: DependencyContainer):
        self.container = container
    
    def extract_content(self, content: str, entry: MemoryEntry) -> List[ContentInfo]:
        media_parts = []
        media_tags = XMLHelper.parse_xml_tag(content, "media_msg")
        
        for attrs, start, end in media_tags:
            if "id" in attrs and attrs["id"] is not None:
                media_id = attrs["id"]
                # 检查媒体ID是否在元数据中
                if "_media_ids" in entry.metadata and media_id in entry.metadata["_media_ids"]:
                    media_parts.append(ContentInfo(
                        content_type="media",
                        start=start,
                        end=end, 
                        text=content[start:end],
                        metadata={"media_id": media_id}
                    ))
        
        return media_parts
    
    def to_llm_content(self, info: ContentInfo) -> LLMChatContentPartType:
        return LLMChatImageContent(media_id=info.metadata["media_id"])
    
    def to_text(self, info: ContentInfo) -> str:
        return f"<media_msg id=\"{info.metadata['media_id']}\" />"


class ToolCallContentStrategy:
    """工具调用内容解析策略"""
    
    def extract_content(self, content: str, entry: MemoryEntry) -> List[ContentInfo]:
        tool_call_parts = []
        tool_call_tags = XMLHelper.parse_xml_tag(content, "function_call")
        
        if "_tool_calls" not in entry.metadata:
            return []
        
        tool_calls = [call for call in entry.metadata["_tool_calls"]]
        
        for attrs, start, end in tool_call_tags:
            if "id" in attrs and attrs["id"] is not None:
                call_id = attrs["id"]
                # 查找对应的工具调用数据
                for call in tool_calls:
                    if call.get("id") == call_id:
                        tool_call_parts.append(ContentInfo(
                            content_type="tool_call",
                            start=start,
                            end=end,
                            text=content[start:end],
                            metadata=call
                        ))
                        break
        
        return tool_call_parts
    
    def to_llm_content(self, info: ContentInfo) -> LLMChatContentPartType:
        return LLMToolCallContent.model_validate(info.metadata)
    
    def to_text(self, info: ContentInfo) -> str:
        return f"<function_call id=\"{info.metadata['id']}\" name=\"{info.metadata['name']}\" />"


class ToolResultContentStrategy:
    """工具结果内容解析策略"""
    
    def extract_content(self, content: str, entry: MemoryEntry) -> List[ContentInfo]:
        tool_result_parts = []
        tool_result_tags = XMLHelper.parse_xml_tag(content, "tool_result")
        
        if "_tool_results" not in entry.metadata:
            return []
        
        tool_results = [result for result in entry.metadata["_tool_results"]]
        
        for attrs, start, end in tool_result_tags:
            if "id" in attrs and attrs["id"] is not None:
                result_id = attrs["id"]
                # 查找对应的工具结果数据
                for result in tool_results:
                    if result.get("id") == result_id:
                        tool_result_parts.append(ContentInfo(
                            content_type="tool_result",
                            start=start,
                            end=end,
                            text=content[start:end],
                            metadata=result
                        ))
                        break
        
        return tool_result_parts
    
    def to_llm_content(self, info: ContentInfo) -> LLMChatContentPartType:
        return LLMToolResultContent.model_validate(info.metadata)
    
    def to_text(self, info: ContentInfo) -> str:
        return f"<tool_result id=\"{info.metadata['id']}\" name=\"{info.metadata['name']}\" isError=\"{info.metadata.get('isError', False)}\" />"


class ContentParser:
    """内容解析器，整合各种内容处理策略"""
    
    def __init__(self, container: DependencyContainer):
        self.container = container
        self.strategies: Dict[str, ContentParseStrategy] = {
            "text": TextContentStrategy(),
            "media": MediaContentStrategy(container),
            "tool_call": ToolCallContentStrategy(),
            "tool_result": ToolResultContentStrategy()
        }
    
    def parse_content(self, content: str, entry: MemoryEntry) -> List[ContentInfo]:
        """解析内容，返回按位置排序的内容信息列表"""
        all_content = []
        
        # 使用所有策略提取内容
        for strategy in self.strategies.values():
            all_content.extend(strategy.extract_content(content, entry))
        
        # 按位置排序
        return sorted(all_content, key=lambda x: x.start)
    
    def to_llm_message(self, content_infos: List[ContentInfo], role: RoleType) -> List[LLMChatMessage]:
        """将内容信息转换为LLM消息"""
        if not content_infos:
            return []
        
        messages = []
        current_content = []
        current_role = role
        
        for info in content_infos:
            strategy = self.strategies.get(info.content_type)
            if not strategy:
                continue
                
            # 对于工具调用和工具结果，创建单独的消息
            if info.content_type == "tool_call":
                # 如果之前有普通内容，先创建一个消息
                if current_content:
                    messages.append(LLMChatMessage(role=current_role, content=current_content))
                    current_content = []
                
                # 创建工具调用消息
                messages.append(LLMChatMessage(
                    role="assistant", 
                    content=[strategy.to_llm_content(info)]
                ))
            elif info.content_type == "tool_result":
                # 如果之前有普通内容，先创建一个消息
                if current_content:
                    messages.append(LLMChatMessage(role=current_role, content=current_content))
                    current_content = []
                
                # 创建工具结果消息
                messages.append(LLMChatMessage(
                    role="tool", 
                    content=[strategy.to_llm_content(info)]
                ))
            else:
                # 普通内容就近拼接
                current_content.append(strategy.to_llm_content(info))
        
        # 处理剩余的普通内容
        if current_content:
            messages.append(LLMChatMessage(role=current_role, content=current_content))
        
        return messages
    
    def to_text(self, content_infos: List[ContentInfo]) -> str:
        """将内容信息转换为文本形式"""
        text_parts = []
        for info in content_infos:
            strategy = self.strategies.get(info.content_type)
            if strategy:
                text_parts.append(strategy.to_text(info))
        return "".join(text_parts)


class DefaultDecomposerStrategy:
    """默认解析策略，将记忆条目转换为文本格式"""
    
    def __init__(self, container: DependencyContainer):
        self.container = container
        self.content_parser = ContentParser(container)
    
    def decompose(self, entries: List[MemoryEntry], context: Dict[str, Any]) -> List[ComposableMessageType]:
        if not entries:
            return [context.get("empty_message", "<空记忆>")]
        
        # 限制最近的条目数量
        entries = entries[-10:]
        
        result: List[ComposableMessageType] = []
        for entry in entries:
            time_diff = datetime.now() - entry.timestamp
            time_str = self._get_time_str(time_diff)
            
            # 解析记忆条目
            content = entry.content or ""
            message_parts = []
            
            if content:
                if "你回答:" in content:
                    # 包含用户消息和AI回答
                    parts = content.split("你回答:", 1)
                    user_content = parts[0].strip()
                    assistant_content = parts[1].strip() if len(parts) > 1 else None
                    
                    # 处理用户消息
                    if user_content:
                        content_infos = self.content_parser.parse_content(user_content, entry)
                        message_parts.append(self.content_parser.to_text(content_infos))
                    
                    # 处理AI回答
                    if assistant_content:
                        content_infos = self.content_parser.parse_content(assistant_content, entry)
                        message_parts.append(f"你回答: {self.content_parser.to_text(content_infos)}")
                else:
                    # 纯用户消息
                    content_infos = self.content_parser.parse_content(content, entry)
                    message_parts.append(self.content_parser.to_text(content_infos))
            
            # 组合所有部分
            result.append(f"{time_str}，{''.join(message_parts)}")
        
        return result
    
    def _get_time_str(self, time_diff: timedelta) -> str:
        """获取时间差的字符串表示"""
        if time_diff.days > 0:
            return f"{time_diff.days}天前"
        elif time_diff.seconds > 3600:
            return f"{time_diff.seconds // 3600}小时前"
        elif time_diff.seconds > 60:
            return f"{time_diff.seconds // 60}分钟前"
        else:
            return "刚刚"


class MultiElementDecomposerStrategy:
    """多元素解析策略，将记忆条目还原为原始对象结构"""
    
    def __init__(self, container: DependencyContainer):
        self.container = container
        self.content_parser = ContentParser(container)
    
    def decompose(self, entries: List[MemoryEntry], context: Dict[str, Any]) -> List[ComposableMessageType]:
        result: List[LLMChatMessage] = []
        
        # 处理每个记忆条目
        for entry in entries:
            messages = self._process_entry(entry)
            result.extend(messages)
        
        # 合并相邻的相同角色消息
        self._merge_adjacent_messages(result)
        
        # 转换为ComposableMessageType类型返回
        return cast(List[ComposableMessageType], result)
    
    def _process_entry(self, entry: MemoryEntry) -> List[LLMChatMessage]:
        """处理单个记忆条目，按照内容顺序解析"""
        result: List[LLMChatMessage] = []
        content = entry.content or ""
        
        if not content:
            return result
            
        if "你回答:" in content:
            # 包含用户消息和AI回答
            parts = content.split("你回答:", 1)
            user_content = parts[0].strip()
            assistant_content = parts[1].strip() if len(parts) > 1 else None
            
            # 处理用户消息
            if user_content:
                content_infos = self.content_parser.parse_content(user_content, entry)
                user_message = self.content_parser.to_llm_message(content_infos, "user")
                if user_message:
                    result.extend(user_message)
            
            # 处理AI回答
            if assistant_content:
                content_infos = self.content_parser.parse_content(assistant_content, entry)
                assistant_message = self.content_parser.to_llm_message(content_infos, "assistant")
                if assistant_message:
                    result.extend(assistant_message)
        else:
            # 纯用户消息
            content_infos = self.content_parser.parse_content(content, entry)
            user_message = self.content_parser.to_llm_message(content_infos, "user")
            if user_message:
                result.extend(user_message)
        
        return result
    
    def _merge_adjacent_messages(self, messages: List[LLMChatMessage]) -> None:
        """合并相邻的相同角色消息"""
        i = 0
        while i < len(messages) - 1:
            current_msg = messages[i]
            next_msg = messages[i + 1]
            
            if current_msg.role == next_msg.role:
                # 合并内容
                current_msg.content.extend(next_msg.content)
                # 删除下一个消息
                messages.pop(i + 1)
            else:
                i += 1 