import base64
from pathlib import Path
from typing import List, Optional, Union

from kirara_ai.media.manager import MediaManager
from kirara_ai.media.metadata import MediaMetadata
from kirara_ai.media.types.media_type import MediaType


class Media:
    """媒体对象，提供更方便的媒体操作接口"""
    
    def __init__(self, media_id: str, media_manager: MediaManager):
        """
        初始化媒体对象
        
        Args:
            media_id: 媒体ID
        """
        self.media_id = media_id
        self._manager = media_manager
        
    @property
    def metadata(self) -> Optional[MediaMetadata]:
        """获取媒体元数据"""
        return self._manager.get_metadata(self.media_id)
    
    @property
    def media_type(self) -> Optional[MediaType]:
        """获取媒体类型"""
        metadata = self.metadata
        return metadata.media_type if metadata else None
    
    @property
    def format(self) -> Optional[str]:
        """获取媒体格式"""
        metadata = self.metadata
        return metadata.format if metadata else None
    
    @property
    def size(self) -> Optional[int]:
        """获取媒体大小"""
        metadata = self.metadata
        return metadata.size if metadata else None
    
    @property
    def description(self) -> Optional[str]:
        """获取媒体描述"""
        metadata = self.metadata
        return metadata.description if metadata else None
    
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
        metadata = self.metadata
        return metadata.mime_type if metadata else None
    
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
    
    async def get_file_path(self) -> Optional[Path]:
        """获取媒体文件路径"""
        return await self._manager.get_file_path(self.media_id)
    
    async def get_data(self) -> Optional[bytes]:
        """获取媒体文件数据"""
        return await self._manager.get_data(self.media_id)
    
    async def get_base64(self) -> Optional[str]:
        """获取媒体文件 base64 编码"""
        data = await self.get_data()
        if data:
            return base64.b64encode(data).decode()
        return None
    
    async def get_url(self) -> Optional[str]:
        """获取媒体文件URL"""
        return await self._manager.get_url(self.media_id)
    
    async def create_message(self):
        """创建媒体消息对象"""
        return await self._manager.create_media_message(self.media_id)
    
    @classmethod
    def from_path(
        cls, 
        path: Union[str, Path], 
        source: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        reference_id: Optional[str] = None,
        manager: Optional[MediaManager] = None
    ) -> 'Media':
        """从文件路径创建媒体对象"""
        media_id = manager.register_from_path(
            str(path),
            source=source,
            description=description,
            tags=tags,
            reference_id=reference_id
        )
        return cls(media_id, manager)
    
    @classmethod
    def from_url(
        cls, 
        url: str, 
        source: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        reference_id: Optional[str] = None,
        manager: Optional[MediaManager] = None
    ) -> 'Media':
        """从URL创建媒体对象"""
        media_id = manager.register_from_url(
            url,
            source=source,
            description=description,
            tags=tags,
            reference_id=reference_id
        )
        return cls(media_id, manager)
    
    @classmethod
    def from_data(
        cls, 
        data: bytes,
        format: Optional[str] = None,
        source: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        reference_id: Optional[str] = None,
        manager: Optional[MediaManager] = None
    ) -> 'Media':
        """从二进制数据创建媒体对象"""
        media_id = manager.register_from_data(
            data,
            format=format,
            source=source,
            description=description,
            tags=tags,
            reference_id=reference_id
        )
        return cls(media_id, manager)
    
    def __str__(self) -> str:
        metadata = self.metadata
        if metadata:
            return f"Media({metadata.media_id}, type={metadata.media_type}, format={metadata.format})"
        return f"Media({self.media_id}, not found)"
    
    def __repr__(self) -> str:
        return self.__str__() 