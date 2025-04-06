from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession, StdioServerParameters, stdio_client, types
from mcp.client.sse import sse_client
from pydantic import AnyUrl

from kirara_ai.config.global_config import MCPServerConfig
from kirara_ai.mcp.models import MCPConnectionState


class MCPServer:
    """
    MCP (Model Control Protocol) 服务器客户端类
    
    用于与 MCP 服务器进行通信，支持 stdio 和 SSE 两种连接模式。
    提供工具调用、补全、资源管理等功能。
    
    本类为 mcp.ClientSession 的代理，
    使其适应 Kirara AI 的生命周期。
    """
    session: Optional[ClientSession] = None
    exit_stack: AsyncExitStack
    state: MCPConnectionState = MCPConnectionState.DISCONNECTED
    
    def __init__(self, server_config: MCPServerConfig):
        """
        初始化 MCP 服务器客户端
        
        Args:
            server_config: MCP 服务器配置
        """
        self.server_config = server_config
        self.exit_stack = AsyncExitStack()
        self.session = None
        self.state = MCPConnectionState.DISCONNECTED
        
    async def connect(self):
        """
        连接到 MCP 服务器
        
        根据配置连接到 MCP 服务器，并初始化会话
        
        Returns:
            bool: 连接是否成功
        """
        if self.state != MCPConnectionState.DISCONNECTED and self.state != MCPConnectionState.ERROR:
            return False
            
        try:
            self.state = MCPConnectionState.CONNECTING
            await self._initialize()
            self.state = MCPConnectionState.CONNECTED
            return True
        except Exception as e:
            self.state = MCPConnectionState.ERROR
            raise e
    
    async def disconnect(self):
        """
        断开与 MCP 服务器的连接
        
        Returns:
            bool: 断开连接是否成功
        """
        if self.state != MCPConnectionState.CONNECTED:
            return False
            
        try:
            self.state = MCPConnectionState.DISCONNECTING
            await self._shutdown()
            self.state = MCPConnectionState.DISCONNECTED
            return True
        except Exception as e:
            self.state = MCPConnectionState.ERROR
            raise e
        
    async def _initialize(self):
        """
        根据配置初始化 MCP 服务器连接
        
        根据 server_config 中的配置选择 stdio 或 SSE 模式进行连接
        """
        if self.server_config.connection_type == "stdio":
            await self._init_stdio_server()
        elif self.server_config.connection_type == "sse":
            await self._init_sse_server()
        else:
            raise ValueError(f"Unsupported server connection type: {self.server_config.connection_type}")

    async def _init_stdio_server(self):
        """初始化 stdio 模式的 MCP 服务器连接"""
        if self.server_config.command is None:
            raise ValueError("command is required in stdio mode")
        
        context = stdio_client(StdioServerParameters(command=self.server_config.command, args=self.server_config.args))
        read, write = await self.exit_stack.enter_async_context(context)
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

    async def _init_sse_server(self):
        """初始化 SSE 模式的 MCP 服务器连接"""
        context = sse_client(self.server_config.url)
        read, write = await self.exit_stack.enter_async_context(context)
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
    
    async def _shutdown(self):
        """关闭 MCP 服务器连接"""
        await self.exit_stack.aclose()
        self.session = None
    
    # 工具相关方法
    
    async def get_tools(self) -> types.ListToolsResult:
        """获取可用工具列表"""
        assert self.session is not None 
        return await self.session.list_tools()
    
    async def call_tool(self, tool_name: str, tool_args: dict) -> types.CallToolResult:
        """
        调用指定工具
        
        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            
        Returns:
            工具调用结果
        """
        assert self.session is not None
        return await self.session.call_tool(tool_name, tool_args)
    
    async def complete(self, prompt: str, tool_args: dict):
        """
        使用模型进行补全
        
        Args:
            prompt: 提示文本
            tool_args: 补全参数
            
        Returns:
            补全结果
        """
        assert self.session is not None
        return await self.session.complete(types.PromptReference(name=prompt, type="ref/prompt"), tool_args)
    
    # 提示词相关方法
    
    async def get_prompt(self, prompt_name: str, prompt_args: dict):
        """
        获取指定提示词
        
        Args:
            prompt_name: 提示词名称
            prompt_args: 提示词参数
            
        Returns:
            提示词内容
        """
        assert self.session is not None
        return await self.session.get_prompt(prompt_name, prompt_args)
    
    async def list_prompts(self):
        """获取可用提示词列表"""
        assert self.session is not None
        return await self.session.list_prompts()
    
    # 资源相关方法
    
    async def list_resources(self):
        """获取可用资源列表"""
        assert self.session is not None
        return await self.session.list_resources()
    
    async def list_resource_templates(self):
        """获取可用资源模板列表"""
        assert self.session is not None
        return await self.session.list_resource_templates()
    
    async def read_resource(self, resource_name: str):
        """
        读取指定资源
        
        Args:
            resource_name: 资源名称
            
        Returns:
            资源内容
        """
        assert self.session is not None
        return await self.session.read_resource(AnyUrl(resource_name))
    
    async def subscribe_resource(self, resource_name: str):
        """
        订阅指定资源
        
        Args:
            resource_name: 资源名称
            
        Returns:
            订阅结果
        """
        assert self.session is not None
        return await self.session.subscribe_resource(AnyUrl(resource_name))
    
    async def unsubscribe_resource(self, resource_name: str):
        """
        取消订阅指定资源
        
        Args:
            resource_name: 资源名称
            
        Returns:
            取消订阅结果
        """
        assert self.session is not None
        return await self.session.unsubscribe_resource(AnyUrl(resource_name))
    
