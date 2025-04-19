from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from kirara_ai.media.types.media_type import MediaType


class MediaMetadata:
    """媒体元数据类"""
    
    def __init__(
        self,
        media_id: str,
        media_type: MediaType,
        format: str,
        size: Optional[int] = None,
        created_at: Optional[datetime] = None,
        source: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        references: Optional[Set[str]] = None,
        url: Optional[str] = None,
        path: Optional[str] = None
    ):
        self.media_id = media_id
        self.media_type = media_type
        self.format = format
        self.size = size
        self.created_at = created_at or datetime.now()
        self.source = source
        self.description = description
        self.tags: List[str] = tags or []
        self.references: Set[str] = references or set()
        self.url = url
        self.path = path
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "media_id": self.media_id,
            "created_at": self.created_at.isoformat(),
            "source": self.source,
            "description": self.description,
            "tags": self.tags,
            "references": list(self.references),
        }
        
        # 添加可选字段
        if self.media_type:
            result["media_type"] = self.media_type.value
        if self.format:
            result["format"] = self.format
        if self.size is not None:
            result["size"] = self.size
        if self.url:
            result["url"] = self.url
        if self.path:
            result["path"] = self.path
            
        return result


    @property
    def mime_type(self) -> str:
        """获取 MIME 类型"""
        return f"{self.media_type.value}/{self.format}"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MediaMetadata':
        """从字典创建元数据"""
        return cls(
            media_id=data["media_id"],
            media_type=MediaType(data["media_type"]),
            format=data["format"],
            size=data.get("size"),
            created_at=datetime.fromisoformat(data["created_at"]),
            source=data.get("source"),
            description=data.get("description"),
            tags=data.get("tags", []),
            references=set(data.get("references", [])),
            url=data.get("url"),
            path=data.get("path")
        ) 