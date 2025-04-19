from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ChatType(Enum):
    C2C = "c2c"
    GROUP = "group"

    @classmethod
    def from_str(cls, value: str) -> "ChatType":
        if value == "c2c" or value == "私聊":
            return cls.C2C
        elif value == "group" or value == "群聊":
            return cls.GROUP
        raise ValueError(f"Invalid chat type: {value}")
    
    def to_str(self) -> str:
        if self == self.C2C:
            return "私聊"
        elif self == self.GROUP:
            return "群聊"
        raise ValueError(f"Invalid chat type: {self}")

@dataclass
class ChatSender:
    """聊天发送者信息封装"""

    display_name: str
    user_id: str
    chat_type: ChatType
    group_id: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    callback = None
    
    @classmethod
    def from_group_chat(
        cls,
        user_id: str,
        group_id: str,
        display_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ChatSender":
        """创建群聊发送者"""
        return cls(
            user_id=user_id,
            chat_type=ChatType.GROUP,
            group_id=group_id,
            display_name=display_name,
            raw_metadata=metadata or {},
        )

    @classmethod
    def from_c2c_chat(
        cls, user_id: str, display_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> "ChatSender":
        """创建私聊发送者"""
        return cls(
            user_id=user_id,
            chat_type=ChatType.C2C,
            display_name=display_name,
            raw_metadata=metadata or {},
        )

    @classmethod
    def get_bot_sender(cls) -> "ChatSender":
        """获取机器人发送者"""
        return cls(
            user_id="bot",
            display_name="bot",
            chat_type=ChatType.C2C,
        )

    def __str__(self) -> str:
        if self.chat_type == ChatType.GROUP:
            return f"{self.group_id}:{self.user_id}"
        else:
            return f"c2c:{self.user_id}"
        
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ChatSender):
            return self.user_id == other.user_id and \
                self.chat_type == other.chat_type and \
                self.group_id == other.group_id
        return False
    
    def __hash__(self) -> int:
        return hash((self.user_id, self.chat_type, self.group_id))
