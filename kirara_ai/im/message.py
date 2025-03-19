from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from kirara_ai.im.sender import ChatSender
from kirara_ai.media import MediaManager, MediaType

MIMETYPE_MAPPING = {
    "image": MediaType.IMAGE,
    "audio": MediaType.AUDIO,
    "video": MediaType.VIDEO,
    "file": MediaType.FILE,
}

# 定义消息元素的基类
class MessageElement(ABC):
    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def to_plain(self):
        pass


# 定义文本消息元素
class TextMessage(MessageElement):
    def __init__(self, text: str):
        self.text = text

    def to_dict(self):
        return {"type": "text", "text": self.text}

    def to_plain(self):
        return self.text

    def __repr__(self):
        return f"TextMessage(text={self.text})"


# 定义媒体消息的基类
class MediaMessage(MessageElement):

    def __init__(
        self,
        url: Optional[str] = None,
        path: Optional[str] = None,
        data: Optional[bytes] = None,
        format: Optional[str] = None,
        media_id: Optional[str] = None,
        reference_id: Optional[str] = None,
        source: Optional[str] = "im_message",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        media_manager: Optional[MediaManager] = None,   
    ):
        self.url = url
        self.path = path
        self.data = data
        self.format = format
        self.media_id = media_id
        self._reference_id = reference_id or f"im_message_{id(self)}"
        self._source = source
        self._description = description
        self._tags = tags or []
        self._media_manager = media_manager or MediaManager()

        # 如果已经有media_id，则直接使用
        if media_id:
            return

        # 注册媒体文件
        self._register_media()

    def _register_media(self) -> None:
        """注册媒体文件"""
        media_manager = self._media_manager
        
        # 根据传入的参数注册媒体文件
        self.media_id = media_manager.register_media(
            url=self.url,
            path=self.path,
            data=self.data,
            format=self.format,
            source=self._source,
            description=self._description,
            tags=self._tags,
            media_type=MIMETYPE_MAPPING[self.resource_type],
            reference_id=self._reference_id
        )
        
        # 获取媒体元数据
        metadata = media_manager.get_metadata(self.media_id)
        if metadata and metadata.format:
            self.format = metadata.format
            if metadata.media_type:
                self.resource_type = metadata.media_type.value

    async def get_url(self) -> str:
        """获取媒体资源的URL"""
        if not self.media_id:
            raise ValueError("Media not registered")
            
        # 如果已经有URL，直接返回
        if self.url:
            return self.url
            
        # 否则从媒体管理器获取
        media_manager = self._media_manager
        url = await media_manager.get_url(self.media_id)
        if url:
            self.url = url  # 缓存结果
            return url
            
        raise ValueError("Failed to get media URL")

    async def get_path(self) -> str:
        """获取媒体资源的文件路径"""
        if not self.media_id:
            raise ValueError("Media not registered")
            
        # 如果已经有路径，直接返回
        if self.path and Path(self.path).exists():
            return self.path
            
        # 否则从媒体管理器获取
        media_manager = self._media_manager
        file_path = await media_manager.get_file_path(self.media_id)
        if file_path:
            self.path = str(file_path)  # 缓存结果
            return self.path
            
        raise ValueError("Failed to get media file path")

    async def get_data(self) -> bytes:
        """获取媒体资源的二进制数据"""
        if not self.media_id:
            raise ValueError("Media not registered")
            
        # 如果已经有数据，直接返回
        if self.data:
            return self.data
            
        # 否则从媒体管理器获取
        media_manager = self._media_manager
        data = await media_manager.get_data(self.media_id)
        if data:
            self.data = data  # 缓存结果
            return data
            
        raise ValueError("Failed to get media data")
    
    def get_description(self) -> str:
        """获取媒体资源的描述"""
        metadata = self._media_manager.get_metadata(self.media_id)
        if metadata:
            return metadata.description
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "type": self.resource_type,
            "media_id": self.media_id,
        }
        
        # 添加可选属性
        if self.format:
            result["format"] = self.format
        if self.url:
            result["url"] = self.url
        if self.path:
            result["path"] = self.path
            
        return result


# 定义语音消息
class VoiceMessage(MediaMessage):
    resource_type = "audio"
    
    def to_dict(self):
        result = super().to_dict()
        result["type"] = "voice"
        return result

    def to_plain(self):
        return "[VoiceMessage]"


# 定义图片消息
class ImageMessage(MediaMessage):
    resource_type = "image"
    
    def to_dict(self):
        result = super().to_dict()
        result["type"] = "image"
        return result

    def to_plain(self):
        return f"[ImageMessage:url={self.url},alt={self.get_description()}]"

    def __repr__(self):
        return f"ImageMessage(media_id={self.media_id}, url={self.url}, path={self.path}, format={self.format})"

# 定义@消息元素
# :deprecated
class AtElement(MessageElement):
    def __init__(self, user_id: str, nickname: str = ""):
        self.user_id = user_id
        self.nickname = nickname

    def to_dict(self):
        return {"type": "at", "data": {"qq": self.user_id, "nickname": self.nickname}}

    def to_plain(self):
        return f"@{self.nickname or self.user_id}"

    def __repr__(self):
        return f"AtElement(user_id={self.user_id}, nickname={self.nickname})"

# 定义@消息元素
class MentionElement(MessageElement):
    def __init__(self, target: ChatSender):
        self.target = target

    def to_dict(self):
        return {"type": "mention", "data": {"target": self.target}}

    def to_plain(self):
        return f"@{self.target.display_name or self.target.user_id}"

    def __repr__(self):
        return f"MentionElement(target={self.target})"

# 定义回复消息元素
class ReplyElement(MessageElement):

    def __init__(self, message_id: str):
        self.message_id = message_id

    def to_dict(self):

        return {"type": "reply", "data": {"id": self.message_id}}

    def to_plain(self):
        return f"[Reply:{self.message_id}]"

    def __repr__(self):
        return f"ReplyElement(message_id={self.message_id})"


# 定义文件消息元素
class FileMessage(MediaMessage):
    resource_type = "file"
    
    def to_dict(self):
        result = super().to_dict()
        result["type"] = "file"
        return result

    def to_plain(self):
        return f"[File:{self.path or self.url or 'unnamed'}]"

    def __repr__(self):
        return f"FileElement(media_id={self.media_id}, url={self.url}, path={self.path}, format={self.format})"


# 定义JSON消息元素
class JsonMessage(MessageElement):

    def __init__(self, data: str):
        self.data = data

    def to_dict(self):
        return {"type": "json", "data": {"data": self.data}}

    def to_plain(self):
        return f"[JSON:{self.data}]"

    def __repr__(self):
        return f"JsonMessage(data={self.data})"


# 定义表情消息元素
class EmojiMessage(MessageElement):

    def __init__(self, face_id: str):
        self.face_id = face_id

    def to_dict(self):

        return {"type": "face", "data": {"id": self.face_id}}

    def to_plain(self):
        return f"[Face:{self.face_id}]"

    def __repr__(self):
        return f"EmojiMessage(face_id={self.face_id})"


# 定义视频消息元素
class VideoMessage(MediaMessage):
    resource_type = "video"
    
    def to_dict(self):
        result = super().to_dict()
        result["type"] = "video"
        return result

    def to_plain(self):
        return f"[Video:{self.path or self.url or 'unnamed'}]"

    def __repr__(self):
        return f"VideoMessage(media_id={self.media_id}, url={self.url}, path={self.path}, format={self.format})"


# 定义消息类
class IMMessage:
    """
    IM消息类，用于表示一条完整的消息。
    包含发送者信息和消息元素列表。

    Attributes:
        sender: 发送者标识
        message_elements: 消息元素列表,可以包含文本、图片、语音等
        raw_message: 原始消息数据
        content: 消息的纯文本内容
        images: 消息中的图片列表
        voices: 消息中的语音列表
    """

    sender: ChatSender
    message_elements: List[MessageElement]
    raw_message: Optional[dict]

    def __repr__(self):
        return f"IMMessage(sender={self.sender}, message_elements={self.message_elements}, raw_message={self.raw_message})"

    @property
    def content(self) -> str:
        """获取消息的纯文本内容"""
        content = ""
        for element in self.message_elements:
            content += element.to_plain()
            if isinstance(element, TextMessage):
                content += "\n"
        return content.strip()

    @property
    def images(self) -> List[ImageMessage]:
        """获取消息中的所有图片"""
        return [
            element
            for element in self.message_elements
            if isinstance(element, ImageMessage)
        ]

    @property
    def voices(self) -> List[VoiceMessage]:
        """获取消息中的所有语音"""
        return [
            element
            for element in self.message_elements
            if isinstance(element, VoiceMessage)
        ]

    def __init__(
        self,
        sender: ChatSender,
        message_elements: List[MessageElement],
        raw_message: dict = None,
    ):
        self.sender = sender
        self.message_elements = message_elements
        self.raw_message = raw_message

    def to_dict(self):
        return {
            "sender": self.sender,
            "message_elements": [
                element.to_dict() for element in self.message_elements
            ],
            "plain_text": "".join(
                [element.to_plain() for element in self.message_elements]
            ),
            "raw_message": self.raw_message,
        }


# backward compatibility
# deprecated
FileElement = FileMessage
ImageElement = ImageMessage
VoiceElement = VoiceMessage
VideoElement = VideoMessage
EmojiElement = EmojiMessage
JsonElement = JsonMessage

# 示例用法
if __name__ == "__main__":
    # 创建消息元素
    text_element = TextMessage("Hello, World!")
    voice_element = VoiceMessage("https://example.com/voice.mp3", 120)
    image_element = ImageMessage("https://example.com/image.jpg", 800, 600)

    # 创建消息对象
    message = IMMessage(
        sender=ChatSender.from_c2c_chat("user123"),
        message_elements=[text_element, voice_element, image_element],
        raw_message={"platform": "example_chat", "timestamp": "2023-10-01T12:00:00Z"},
    )

    # 转换为字典格式
    message_dict = message.to_dict()
    print(message_dict)
