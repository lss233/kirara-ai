import re
from datetime import datetime, timedelta
from typing import List, Optional, Union

from kirara_ai.im.message import IMMessage, MediaMessage, TextMessage
from kirara_ai.im.sender import ChatSender
from kirara_ai.llm.format.message import (LLMChatContentPartType, LLMChatImageContent, LLMChatMessage, LLMToolCallContent,
                                          LLMToolResultContent, LLMChatTextContent, RoleType)
from kirara_ai.logger import get_logger
from kirara_ai.media.manager import MediaManager
from kirara_ai.memory.entry import MemoryEntry

from .base import ComposableMessageType, MemoryComposer, MemoryDecomposer


def drop_think_part(text: str) -> str:
    return re.sub(r"(?:<think>[\s\S]*?</think>)?([\s\S]*)", r"\1", text, flags=re.DOTALL)

class DefaultMemoryComposer(MemoryComposer):
    def compose(
        self, sender: Optional[ChatSender], message: List[ComposableMessageType]
    ) -> MemoryEntry:
        composed_message = ""
        media_ids = []
        tool_calls = []
        tool_results = []
        for msg in message:
            if isinstance(msg, IMMessage):
                composed_message += f"{msg.sender.display_name} 说: \n"
                for element in msg.message_elements:
                    if isinstance(element, MediaMessage):
                        desc = element.get_description()
                        composed_message += f"<media_msg id={element.media_id} desc=\"{desc}\" />\n"
                        media_ids.append(element.media_id)
                    elif isinstance(element, TextMessage):
                        composed_message += f"{element.to_plain()}\n"
                    else:
                        composed_message += element.to_plain()
            elif isinstance(msg, LLMChatMessage):
                temp = ""
                for part in msg.content:
                    if isinstance(part, LLMChatTextContent):
                        temp += f"{drop_think_part(part.text)}\n"
                    elif isinstance(part, LLMChatImageContent):
                        media = self.container.resolve(MediaManager).get_media(part.media_id)
                        desc = media.description if media else ""
                        temp += f"<media_msg id={part.media_id} desc=\"{desc}\" />\n"
                        media_ids.append(part.media_id)
                    elif isinstance(part, LLMToolCallContent):
                        # 将 工具类转化为json字符串存储减少空间占用
                        tool_calls.append(part.model_dump_json())
                    elif isinstance(part, LLMToolResultContent):
                        tool_results.append(part.model_dump_json())
                if temp.strip("\n"): 
                    # 防止空消息,
                    # 根据四大adapter的返回结果，LLMChatResponse 返回的是tool_call时, 其content必定为LLMToolCallContent + TextContent的深度思考内容(仅来源于claude).
                    # 所以为了不干扰后续解压功能这里使用temp接收message，当temp去除头尾换行符后不为空时才添加到composed_message中
                    composed_message += f"你回答: \n" + temp

        composed_message = composed_message.strip()
        composed_at = datetime.now()
        return MemoryEntry(
            sender=sender or ChatSender.get_bot_sender(),
            content=composed_message,
            timestamp=composed_at,
            metadata={
                "_media_ids": media_ids,
                "_tool_calls": tool_calls,
                "_tool_results": tool_results,
            },
        )


class DefaultMemoryDecomposer(MemoryDecomposer):
    def decompose(self, entries: List[MemoryEntry]) -> List[ComposableMessageType]:
        if len(entries) == 0:
            return [self.empty_message]

        # 7秒前，<记忆内容>
        memory_texts: List[ComposableMessageType] = []
        for entry in entries[-10:]:
            time_diff = datetime.now() - entry.timestamp
            time_str = self.get_time_str(time_diff)
            memory_texts.append(f"{time_str}，{entry.content}")

        return memory_texts

    def get_time_str(self, time_diff: timedelta) -> str:
        if time_diff.days > 0:
            return f"{time_diff.days}天前"
        elif time_diff.seconds > 3600:
            return f"{time_diff.seconds // 3600}小时前"
        elif time_diff.seconds > 60:
            return f"{time_diff.seconds // 60}分钟前"
        else:
            return "刚刚"


class MultiElementDecomposer(MemoryDecomposer):
    logger = get_logger("MultiElementDecomposer")
    
    def decompose(self, entries: List[MemoryEntry]) -> List[LLMChatMessage]:
        decomposed_messages = []
        for entry in entries:
            if not entry.content:
                # content为空字符串， 证明其为工具消息
                content: list[LLMChatContentPartType] = []
                if tool_calls := entry.metadata.get("_tool_calls"):
                    for call in tool_calls:
                        tool_call = LLMToolCallContent.model_validate_json(call)
                        content.append(tool_call)
                    decomposed_messages.append(LLMChatMessage(role="assistant", content=content))
                elif tool_results := entry.metadata.get("_tool_results"):
                    for result in tool_results:
                        tool_result = LLMToolResultContent.model_validate_json(result)
                        content.append(tool_result)
                    decomposed_messages.append(LLMChatMessage(role="tool", content=content))
            # 判断MemoryEntry的内容是否包含"你回答:"
            elif "你回答:" in entry.content:
                # 如果包含，则分割MemoryEntry的内容为用户消息和AI回答
                user_content = entry.content.split("你回答:")[0].strip()
                assistant_content = entry.content.split("你回答:")[1].strip()

                # 处理用户消息
                user_message = self.create_llm_chat_message(user_content, "user", entry.sender)
                if user_message:
                    decomposed_messages.append(user_message)

                # 处理AI回答
                assistant_message = self.create_llm_chat_message(assistant_content, "assistant", entry.sender)
                if assistant_message:
                    decomposed_messages.append(assistant_message)
            else:
                # 如果不包含"你回答:"，则全部视为用户消息
                user_message = self.create_llm_chat_message(entry.content, "user", entry.sender)
                if user_message:
                    decomposed_messages.append(user_message)

        # 合并相邻的相同角色消息
        i = 0
        while i < len(decomposed_messages) - 1:
            current_msg = decomposed_messages[i]
            next_msg = decomposed_messages[i + 1]
            
            if current_msg.role == next_msg.role:
                # 合并内容
                current_msg.content.extend(next_msg.content)
                # 删除下一个消息
                decomposed_messages.pop(i + 1)
            else:
                i += 1

        return decomposed_messages

    def create_llm_chat_message(self, content: str, role: RoleType, sender: ChatSender) -> Union[LLMChatMessage, None]:
        message_content: List[LLMChatContentPartType] = []

        # 使用正则表达式提取 <media_msg> 标签
        media_msg_pattern = re.compile(r'<media_msg id=(.*?) desc="(.*?)" />')
        matches = media_msg_pattern.findall(content)

        last_index = 0
        for media_id, description in matches:
            start_index = content.find(f'<media_msg id={media_id} desc="{description}" />', last_index)
            if start_index > last_index:
                text_content = content[last_index:start_index]
                message_content.append(LLMChatTextContent(text=text_content))
            # 校验媒体资源有效性
            media_object = self.container.resolve(MediaManager).get_media(media_id)
            if media_object:
                message_content.append(LLMChatImageContent(media_id=media_id))
            else:
                self.logger.warning(f"媒体资源无效: {media_id}")
                message_content.append(LLMChatTextContent(text=f"<media_msg id={media_id} desc=\"{description}\" status=\"error: 媒体资源无效\"/> "))
            last_index = start_index + len(f'<media_msg id={media_id} desc="{description}" />')

        # 处理剩余的文本内容
        if last_index < len(content):
            message_content.append(LLMChatTextContent(text=content[last_index:]))

        if message_content:
            return LLMChatMessage(role=role, content=message_content)
        else:
            return None

