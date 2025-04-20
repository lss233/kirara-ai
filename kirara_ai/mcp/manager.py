#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
from typing import Dict, NamedTuple, Optional, Tuple

from mcp import types

from kirara_ai.config.global_config import GlobalConfig, MCPServerConfig
from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.logger import get_logger
from kirara_ai.mcp.models import MCPConnectionState
from kirara_ai.mcp.server import MCPServer

logger = get_logger("MCP")

class ToolCacheEntry(NamedTuple):
    """工具缓存条目"""
    server_id: str          # 服务器ID
    original_name: str      # 原始工具名称
    tool_info: types.Tool  # 工具信息

class MCPServerManager:
    """MCP服务器管理器，负责管理和控制MCP服务器进程"""

    def __init__(self, container: DependencyContainer):
        """初始化MCP服务器管理器"""
        self.container = container
        self.config = container.resolve(GlobalConfig)
        self.servers: Dict[str, MCPServer] = {}
        self.tools_cache: Dict[str, ToolCacheEntry] = {}
            
    def load_servers(self):
        """从配置加载所有MCP服务器"""
        for server_config in self.config.mcp.servers:
            try:
                self.load_server(server_config)
            except Exception as e:
                logger.opt(exception=e).error(f"Failed to load MCP server {server_config.id}")
        logger.info(f"MCP server manager initialized, loaded {len(self.servers)} servers")
        
    def load_server(self, server_config: MCPServerConfig) -> MCPServer:
        """从配置加载MCP服务器"""
        server = MCPServer(server_config)
        logger.info(f"Initializing MCP server {server_config.id}")
        self.servers[server_config.id] = server
        return server
    
    def get_all_servers(self) -> Dict[str, MCPServer]:
        """获取所有MCP服务器列表"""
        return self.servers
    
    def get_server(self, server_id: str) -> Optional[MCPServer]:
        """获取指定ID的MCP服务器"""
        return self.servers.get(server_id)
    
    def is_server_id_available(self, server_id: str) -> bool:
        """
        检查服务器ID是否可用
        
        判断条件：
        1. 服务器ID不存在
        2. 或者服务器存在但状态为 DISCONNECTED 或 ERROR
        """
        if server_id not in self.servers:
            return True
        
        server = self.servers[server_id]
        return server.state in [MCPConnectionState.DISCONNECTED, MCPConnectionState.ERROR]
    
    def get_statistics(self) -> Dict[str, int]:
        """获取MCP服务器统计信息"""
        total = len(self.servers)
        stdio = sum(bool(s.server_config.connection_type == "stdio") for s in self.servers.values())
        sse = sum(bool(s.server_config.connection_type == "sse") for s in self.servers.values())
        connected = sum(bool(s.state == MCPConnectionState.CONNECTED) for s in self.servers.values())
        disconnected = sum(bool(s.state == MCPConnectionState.DISCONNECTED) for s in self.servers.values())
        error = sum(bool(s.state == MCPConnectionState.ERROR) for s in self.servers.values())
        
        return {
            "total": total,
            "stdio": stdio,
            "sse": sse,
            "connected": connected,
            "disconnected": disconnected,
            "error": error
        }
    def connect_all_servers(self, loop: asyncio.AbstractEventLoop):
        """连接所有MCP服务器"""
        async def _connect_server_safe(server_id):
            try:
                await self.connect_server(server_id)
            except Exception as e:
                logger.opt(exception=e).error(f"Exception occurred when connecting MCP server {server_id}")
        
        tasks = []
        for server_id in self.servers.keys():
            task = loop.create_task(_connect_server_safe(server_id))
            tasks.append(task)
            
        if tasks:
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            
    async def connect_server(self, server_id: str) -> bool:
        """连接MCP服务器"""
        server = self.servers.get(server_id)
        if not server:
            logger.error(f"Cannot connect to non-existent MCP server: {server_id}")
            return False
        
        if server.state == MCPConnectionState.CONNECTED:
            logger.warning(f"MCP server {server_id} is already connected")
            return True
        
        try:
            logger.info(f"Connecting to MCP server {server_id}")
            
            # 连接到服务器
            success = await server.connect()
            
            if not success:
                logger.error(f"Failed to connect to MCP server {server_id}")
                return False
            
            # 连接成功后，更新工具缓存
            await self._update_tools_cache(server_id)
            
            logger.info(f"Successfully connected to MCP server {server_id}")
            return True
            
        except Exception as e:
            logger.opt(exception=e).error(f"Error occurred when connecting to MCP server {server_id}")
            return False
    
    def disconnect_all_servers(self, loop: asyncio.AbstractEventLoop):
        """断开所有MCP服务器连接"""
        disconnect_tasks = []
        for server_id, server in self.servers.items():
            if server.state == MCPConnectionState.CONNECTED:
                disconnect_tasks.append(loop.create_task(self.stop_server(server_id)))
        
        if disconnect_tasks:
            loop.run_until_complete(asyncio.gather(*disconnect_tasks, return_exceptions=True))
            
        self.tools_cache.clear()
        
        logger.info("All MCP servers have been disconnected")
            
    async def stop_server(self, server_id: str) -> bool:
        """断开MCP服务器连接"""
        server = self.servers.get(server_id)
        
        if not server:
            logger.error(f"Cannot disconnect from non-existent MCP server: {server_id}")
            return False
        
        if server.state != MCPConnectionState.CONNECTED:
            logger.warning(f"MCP server {server_id} is not connected")
            return True
        
        try:
            logger.info(f"Disconnecting from MCP server {server_id}")
            
            # 断开服务器连接
            success = await server.disconnect()
            
            if not success:
                logger.error(f"Failed to disconnect from MCP server {server_id}")
                return False
            
            # 从工具缓存中移除该服务器的工具
            self._remove_server_tools_from_cache(server_id)
            
            logger.info(f"Successfully disconnected from MCP server {server_id}")
            return True
            
        except Exception as e:
            logger.opt(exception=e).error(f"Error occurred when disconnecting from MCP server {server_id}")
            return False
    
    async def _update_tools_cache(self, server_id: str) -> bool:
        """
        更新指定服务器的工具缓存
        
        Args:
            server_id: 服务器ID
            
        Returns:
            bool: 更新是否成功
        """
        server = self.servers.get(server_id)
        if not server or server.state != MCPConnectionState.CONNECTED:
            return False
        
        try:
            # 获取服务器工具列表
            tools = await server.get_tools()
            
            # 先移除该服务器的旧工具
            self._remove_server_tools_from_cache(server_id)
            # 添加新工具到缓存
            for tool in tools.tools:
                original_name = tool.name
                if not original_name:
                    continue
                
                # 检查工具名称是否已存在
                if original_name in self.tools_cache:
                    # 名称冲突，使用 server_id.tool_name 作为新名称
                    display_name = f"{server.server_config.id}.{original_name}"
                    logger.warning(f"工具名称冲突: {original_name}，重命名为 {display_name}")
                else:
                    display_name = original_name
                
                # 存储工具信息
                self.tools_cache[display_name] = ToolCacheEntry(
                    server_id=server_id,
                    original_name=original_name,
                    tool_info=tool
                )
            
            return True
        except Exception as e:
            logger.opt(exception=e).error(f"更新服务器 {server_id} 工具缓存时发生错误")
            return False
    
    def _remove_server_tools_from_cache(self, server_id: str):
        """
        从工具缓存中移除指定服务器的所有工具
        
        Args:
            server_id: 服务器ID
        """
        # 找出属于该服务器的所有工具名称
        tool_names_to_remove = [
            name for name, entry in self.tools_cache.items() if entry.server_id == server_id
        ]
        
        # 从缓存中移除这些工具
        for name in tool_names_to_remove:
            self.tools_cache.pop(name, None)
    
    def get_tools(self) -> Dict[str, ToolCacheEntry]:
        """
        获取所有可用工具
        """
        # 返回工具信息
        return self.tools_cache
    
    def get_tool_server(self, tool_name: str) -> Optional[Tuple[MCPServer, str]]:
        """
        根据工具名称获取对应的服务器实例和原始工具名称
        
        Args:
            tool_name: 工具显示名称
            
        Returns:
            Optional[Tuple[MCPServer, str]]: (服务器实例, 原始工具名称)，如果工具不存在则返回None
        """
        if tool_name not in self.tools_cache:
            return None
        
        entry = self.tools_cache[tool_name]
        server = self.servers.get(entry.server_id)
        if not server:
            return None
        
        return (server, entry.original_name)
    
    async def call_tool(self, tool_name: str, tool_args: dict) -> Optional[types.CallToolResult]:
        """
        调用指定工具
        
        Args:
            tool_name: 工具显示名称
            tool_args: 工具参数
            
        Returns:
            Optional[dict]: 工具调用结果，如果调用失败则返回None
        """
        result = self.get_tool_server(tool_name)
        if not result:
            logger.error(f"Tool {tool_name} not found or server not available")
            return None
        
        server, original_name = result
        
        if server.state != MCPConnectionState.CONNECTED:
            logger.error(f"Server for tool {tool_name} is not connected")
            return None
        
        try:
            # 使用原始工具名称调用
            call_tool_result = await server.call_tool(original_name, tool_args)
            return call_tool_result
        except Exception as e:
            logger.opt(exception=e).error(f"Error occurred when calling tool {tool_name}")
            return None
