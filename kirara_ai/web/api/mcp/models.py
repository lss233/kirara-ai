from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MCPServerInfo(BaseModel):
    """MCP服务器信息"""
    id: str
    description: Optional[str] = None
    connection_type: str
    command: Optional[str] = None
    args: str = Field(default="")
    url: Optional[str] = None
    connection_state: str

class MCPToolInfo(BaseModel):
    """MCP工具信息"""
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)

class MCPServerList(BaseModel):
    """MCP服务器列表"""
    items: List[MCPServerInfo]
    total: int
    page: int
    page_size: int
    total_pages: int


class MCPStatistics(BaseModel):
    """MCP统计信息"""
    total_servers: int
    stdio_servers: int
    sse_servers: int
    connected_servers: int
    disconnected_servers: int
    error_servers: int
    total_tools: int


class MCPServerCreateRequest(BaseModel):
    """创建MCP服务器请求"""
    id: str
    description: Optional[str] = None
    command: str
    args: str
    connection_type: str


class MCPServerUpdateRequest(BaseModel):
    """更新MCP服务器请求"""
    description: Optional[str] = None
    command: Optional[str] = None
    args: str = Field(default="")
    connection_type: Optional[str] = None
    url: Optional[str] = None