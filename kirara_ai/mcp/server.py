import asyncio
import anyio
from contextlib import AsyncExitStack
from typing import Optional

import anyio.lowlevel
from mcp import ClientSession, StdioServerParameters, stdio_client, types
from mcp.client.sse import sse_client
from mcp.shared.session import RequestResponder
from pydantic import AnyUrl

from kirara_ai.config.global_config import MCPServerConfig
from kirara_ai.logger import get_logger
from kirara_ai.mcp.models import MCPConnectionState
from .manager import MCPServerManager

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
    manager: MCPServerManager
    def __init__(self, server_config: MCPServerConfig, manager: MCPServerManager):
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
        self.manager = manager
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
                    args=self.server_config.args,
                    env=self.server_config.env
                ))
            elif self.server_config.connection_type == "sse":
                self._client = sse_client(self.server_config.url)
            else:
                raise ValueError(f"不支持的服务器连接类型: {self.server_config.connection_type}")
            
            # 使用 exit_stack 管理资源
            read, write = await exit_stack.enter_async_context(self._client)
            self.session = await exit_stack.enter_async_context(ClientSession(read, write, message_handler=self.message_handler_callback))
            
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
    
    async def message_handler_callback(
            self,
            req: RequestResponder[types.ServerRequest, types.ClientResult]
            | types.ServerNotification
            | Exception,
        ) -> None:
        """
        按照sdk写法添加了针对Server通知的处理
        虽然ClientSession中有只处理SeverNotification的函数，但是sdk没有暴露接收这个函数的参数。
        所以只能用这个更大范围的。除非你重写ClientSession._received_notification函数。
        或者能水个pr?

        Args:
            req: 请求响应器或通知或异常
        """
        # ServerNotification是一个联合类型，自己查看有哪些
        if isinstance(req, types.ToolListChangedNotification):
            # 处理工具变更信息
            self.manager._update_tools_cache(self.server_config.id)
        elif isinstance(req, types.PromptListChangedNotification):
            # 处理提示词变更信息
            self.manager._update_prompts_cache(self.server_config.id)
        elif isinstance(req, types.ResourceListChangedNotification):
            # 处理资源变更信息
            self.manager._update_resources_cache(self.server_config.id)
        else:
            # 处理其他通知
            logger.warning(f"MCP客户端接收到服务器{self.server_config.id}的通知，但未对其进行处理: {req}")

        # 交还控制权给loop
        anyio.lowlevel.checkpoint()

    async def list_client_roots_callback(ctx) -> types.ListRootsResult:
        """
        列出客户端允许的资源根目录

        Args:

        Returns:
            types.ListRootsResult: 资源根目录列表
        """
        # 这个需要kirara-agent做出较大支持，webApi 中设定允许的资源根，最好弄个单独的目录。
        # 文件根格式为file:///myResource/, 也可以为一个url.
        pass

    async def send_ping(self) -> None:
        await self.session.send_ping()

    async def send_notification(self, notification: types.ClientNotification) -> None:
        """
        给服务器发消息，例如资源根更改
        不使用ClientSession的send_roots_list_changed，因为它只支持发送RootsListChangedNotification。
        这里使用其父对象BaseSession的send_notification, 其支持发送所有ClientNotification。
        ps: 这难道不是双工吗？

        Args:
            notification: 客户端通知
        """
        assert self.session is not None
        await self.session.send_notification(notification)

    async def sampling_callback():
        """
        采样回调函数
        写这只是为了提醒你支持这个回调函数，现在不实现。等出现未知bug就应该使用这个了。相当于mcp debug？
        """
        pass
    async def logging_callback():
        """
        日志回调函数
        写这只是为了提醒你支持这个回调函数，现在不实现。
        ps: 居然支持自定义日志
        """
        pass