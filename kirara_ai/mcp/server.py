import asyncio
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession, StdioServerParameters, stdio_client, types
from mcp.client.sse import sse_client
from pydantic import AnyUrl

from kirara_ai.config.global_config import MCPServerConfig
from kirara_ai.logger import get_logger
from kirara_ai.mcp.models import MCPConnectionState

logger = get_logger("MCP.Server")

class MCPServer:
    """
    MCP (Model Control Protocol) 服务器客户端类
    
    用于与 MCP 服务器进行通信，支持 stdio 和 SSE 两种连接模式。
    提供工具调用、补全、资源管理等功能。
    
    本类为 mcp.ClientSession 的代理，
    使其适应 Kirara AI 的生命周期。
    """
    session: Optional[ClientSession] = None
    state: MCPConnectionState = MCPConnectionState.DISCONNECTED
    
    def __init__(self, server_config: MCPServerConfig):
        """
        初始化 MCP 服务器客户端
        
        Args:
            server_config: MCP 服务器配置
        """
        self.server_config = server_config
        self.session = None
        self.state = MCPConnectionState.DISCONNECTED
        self._lifecycle_task = None
        self._shutdown_event = asyncio.Event()
        self._connected_event = asyncio.Event()
        self._client = None
        
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
            
            # 重置事件
            self._shutdown_event.clear()
            self._connected_event.clear()
            
            # 创建并启动生命周期任务
            if self._lifecycle_task is None or self._lifecycle_task.done():
                self._lifecycle_task = asyncio.create_task(self._lifecycle_manager())
            
            # 等待连接完成或超时
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=30.0)
                if self.state != MCPConnectionState.CONNECTED:
                    # 连接失败
                    return False
                return True
            except asyncio.TimeoutError:
                logger.error(f"连接到 MCP 服务器 {self.server_config.id} 超时")
                await self.disconnect()  # 超时时断开连接
                return False
                
        except Exception as e:
            self.state = MCPConnectionState.ERROR
            logger.opt(exception=e).error(f"连接 MCP 服务器 {self.server_config.id} 时发生错误")
            return False
    
    async def disconnect(self):
        """
        断开与 MCP 服务器的连接
        
        Returns:
            bool: 断开连接是否成功
        """
        if self.state == MCPConnectionState.DISCONNECTED:
            return True
            
        try:
            self.state = MCPConnectionState.DISCONNECTING
            
            # 发送关闭信号
            self._shutdown_event.set()
            
            # 等待生命周期任务完成
            if self._lifecycle_task and not self._lifecycle_task.done():
                try:
                    await asyncio.wait_for(self._lifecycle_task, timeout=10.0)
                except asyncio.TimeoutError:
                    # 如果任务没有及时完成，取消它
                    self._lifecycle_task.cancel()
                    try:
                        await self._lifecycle_task
                    except (asyncio.CancelledError, Exception):
                        pass
            
            self.state = MCPConnectionState.DISCONNECTED
            return True
        except Exception as e:
            self.state = MCPConnectionState.ERROR
            logger.opt(exception=e).error(f"断开 MCP 服务器 {self.server_config.id} 连接时发生错误")
            return False
        
    async def _lifecycle_manager(self):
        """
        服务器生命周期管理任务
        
        负责服务器的连接、运行和断开连接的完整生命周期
        """
        exit_stack = AsyncExitStack()
        
        try:
            # 初始化连接
            if self.server_config.connection_type == "stdio":
                self._client = stdio_client(StdioServerParameters(
                    command=self.server_config.command, 
                    args=self.server_config.args
                ))
            elif self.server_config.connection_type == "sse":
                self._client = sse_client(self.server_config.url)
            else:
                raise ValueError(f"不支持的服务器连接类型: {self.server_config.connection_type}")
            
            # 使用 exit_stack 管理资源
            read, write = await exit_stack.enter_async_context(self._client)
            self.session = await exit_stack.enter_async_context(ClientSession(read, write))
            
            # 初始化会话
            await self.session.initialize()
            
            # 更新状态并通知连接完成
            self.state = MCPConnectionState.CONNECTED
            self._connected_event.set()
            
            # 等待关闭信号
            await self._shutdown_event.wait()
            
        except Exception as e:
            # 连接失败
            self.state = MCPConnectionState.ERROR
            self._connected_event.set()  # 通知连接过程已完成（虽然是失败的）
            logger.opt(exception=e).error(f"MCP server {self.server_config.id} lifecycle task error")
        finally:
            # 清理资源
            self.session = None
            self._client = None
            
            # 关闭所有资源
            await exit_stack.aclose()
            
            # 如果状态仍然是 DISCONNECTING，则更新为 DISCONNECTED
            if self.state == MCPConnectionState.DISCONNECTING:
                self.state = MCPConnectionState.DISCONNECTED

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
    
