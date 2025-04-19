import base64
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from kirara_ai.im.message import MediaMessage
from kirara_ai.media.manager import MediaManager
from kirara_ai.media.metadata import MediaMetadata
from kirara_ai.media.types.media_type import MediaType


class Media:
    """媒体对象，提供更方便的媒体操作接口"""
    
    metadata: MediaMetadata
    
    def __init__(self, media_id: str, media_manager: MediaManager):
        """
        初始化媒体对象
        
        Args:
            media_id: 媒体ID
        """
        self.media_id = media_id
        self._manager = media_manager
        metadata = self._manager.get_metadata(self.media_id)
        assert metadata is not None, f"Media metadata not found for {self.media_id}"
        self.metadata = metadata
    
    @property
    def media_type(self) -> MediaType:
        """获取媒体类型"""
        return self.metadata.media_type
    
    @property
    def format(self) -> str:
        """获取媒体格式"""
        return self.metadata.format
    
    @property
    def size(self) -> Optional[int]:
        """获取媒体大小"""
        return self.metadata.size
    
    @property
    def description(self) -> Optional[str]:
        """获取媒体描述"""
        return self.metadata.description
    
    @description.setter
    def description(self, value: str) -> None:
        """设置媒体描述"""
        self._manager.update_metadata(self.media_id, description=value)
    
    @property
    def tags(self) -> List[str]:
        """获取媒体标签"""
        metadata = self.metadata
        return metadata.tags if metadata else []
    
    @property
    def mime_type(self) -> str:
        """获取媒体 MIME 类型"""
        return self.metadata.mime_type
    
    def add_tags(self, tags: List[str]) -> None:
        """添加标签"""
        self._manager.add_tags(self.media_id, tags)
    
    def remove_tags(self, tags: List[str]) -> None:
        """移除标签"""
        self._manager.remove_tags(self.media_id, tags)
    
    def add_reference(self, reference_id: str) -> None:
        """添加引用"""
        self._manager.add_reference(self.media_id, reference_id)
    
    def remove_reference(self, reference_id: str) -> None:
        """移除引用"""
        self._manager.remove_reference(self.media_id, reference_id)
    
    async def get_file_path(self) -> Path:
        """获取媒体文件路径"""
        path = await self._manager.get_file_path(self.media_id)
        assert path is not None, f"Media file path not found for {self.media_id}"
        return path
    
    async def get_data(self) -> bytes:
        """获取媒体文件数据"""
        data = await self._manager.get_data(self.media_id)
        assert data is not None, f"Media data not found for {self.media_id}"
        return data
    
    async def get_base64(self) -> str:
        """获取媒体文件 base64 编码"""
        data = await self.get_data()
        assert data is not None, "Media data is None"
        return base64.b64encode(data).decode()
    
    async def get_url(self) -> str:
        """获取媒体文件URL"""
        url = await self._manager.get_url(self.media_id)
        assert url is not None, f"Media URL not found for {self.media_id}"
        return url
    
    async def get_base64_url(self) -> str:
        """获取媒体文件 base64 URL"""
        return f"data:{self.mime_type};base64,{await self.get_base64()}"
    
    async def create_message(self) -> "MediaMessage":
        """创建媒体消息对象"""
        message = await self._manager.create_media_message(self.media_id)
        assert message is not None, f"Media message not found for {self.media_id}"
        return message
    
    def __str__(self) -> str:
        metadata = self.metadata
        if metadata:
            return f"Media({metadata.media_id}, type={metadata.media_type}, format={metadata.format})"
        return f"Media({self.media_id}, not found)"
    
    def __repr__(self) -> str:
        return self.__str__() 