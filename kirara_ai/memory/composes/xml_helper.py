import re
from typing import Dict, List, Optional, Tuple


class XMLHelper:
    """XML 格式化和解析的辅助工具类"""

    @staticmethod
    def escape_xml_attr(text: str) -> str:
        """转义XML属性中的特殊字符"""
        if not isinstance(text, str):
            text = str(text)
        return text.replace("&", "&amp;").replace("\"", "&quot;").replace("<", "&lt;").replace(">", "&gt;")

    @staticmethod
    def unescape_xml_attr(text: str) -> str:
        """反转义XML属性中的特殊字符"""
        return text.replace("&quot;", "\"").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

    @staticmethod
    def create_xml_tag(tag_name: str, attributes: Dict[str, Optional[str]], self_closing: bool = True) -> str:
        """创建XML标签，支持 null safety（None 值的属性将被忽略）"""
        attrs_str = " ".join([f'{k}="{XMLHelper.escape_xml_attr(v)}"' for k, v in attributes.items() if v is not None])
        if self_closing:
            return f"<{tag_name} {attrs_str} />"
        else:
            return f"<{tag_name} {attrs_str}>"

    @staticmethod
    def parse_xml_tag(content: str, tag_name: str) -> List[Tuple[Dict[str, Optional[str]], int, int]]:
        """解析XML标签，返回属性字典和标签在原文中的起始、结束位置
        如果属性在标签中不存在，则在返回的字典中该属性值为 None
        """
        pattern = re.compile(f'<{tag_name}\\s+(.*?)\\s*/>')
        attr_pattern = re.compile(r'(\w+)="(.*?)"')
        
        results: List[Tuple[Dict[str, Optional[str]], int, int]] = []
        for match in pattern.finditer(content):
            attrs_text = match.group(1)
            attrs: Dict[str, Optional[str]] = {name: XMLHelper.unescape_xml_attr(value) for name, value in attr_pattern.findall(attrs_text)}
            results.append((attrs, match.start(), match.end()))
        
        return results

    @staticmethod
    def get_attr(attrs: Dict[str, Optional[str]], key: str, default: Optional[str] = None) -> Optional[str]:
        """安全地从属性字典中获取值，如果不存在则返回默认值"""
        return attrs.get(key, default) 