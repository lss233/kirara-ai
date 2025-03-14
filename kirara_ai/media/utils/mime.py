from typing import Tuple

import magic

from kirara_ai.media.types.media_type import MediaType

# MIME类型重映射
mime_remapping = {
    "audio/mpeg": "audio/mp3",
    "audio/x-wav": "audio/wav",
    "audio/x-m4a": "audio/m4a",
    "audio/x-flac": "audio/flac",
}


def detect_mime_type(data: bytes = None, path: str = None) -> Tuple[str, MediaType, str]:
    """
    检测文件的MIME类型
    
    Args:
        data: 文件数据
        path: 文件路径
        
    Returns:
        Tuple[str, MediaType, str]: (mime_type, media_type, format)
    """
    if data is not None:
        mime_type = magic.from_buffer(data, mime=True)
    elif path is not None:
        mime_type = magic.from_file(path, mime=True)
    else:
        raise ValueError("Must provide either data or path")
    
    if mime_type in mime_remapping:
        mime_type = mime_remapping[mime_type]
        
    media_type = MediaType.from_mime(mime_type)
    format = mime_type.split('/')[-1]
    
    return mime_type, media_type, format 