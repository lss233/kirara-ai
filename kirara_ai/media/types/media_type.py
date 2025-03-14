from enum import Enum


class MediaType(Enum):
    """媒体类型枚举"""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    
    @classmethod
    def from_mime(cls, mime_type: str) -> 'MediaType':
        """从MIME类型获取媒体类型"""
        main_type = mime_type.split('/')[0]
        if main_type == "image":
            return cls.IMAGE
        elif main_type == "audio":
            return cls.AUDIO
        elif main_type == "video":
            return cls.VIDEO
        else:
            return cls.FILE 