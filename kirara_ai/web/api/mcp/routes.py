#!/usr/bin/env python
# -*- coding: utf-8 -*-


from quart import Blueprint, g, jsonify, request

from kirara_ai.config.config_loader import CONFIG_FILE, ConfigLoader
from kirara_ai.config.global_config import GlobalConfig, MCPServerConfig
from kirara_ai.logger import get_logger
from kirara_ai.mcp import MCPConnectionState, MCPServer, MCPServerManager

from ...auth.middleware import require_auth
from .models import (MCPServerCreateRequest, MCPServerInfo, MCPServerList, MCPServerUpdateRequest, MCPStatistics,
                     MCPToolInfo)

# 创建蓝图
mcp_bp = Blueprint("mcp", __name__)
logger = get_logger("WebServer.MCP")


def _convert_to_server_info(server: MCPServer) -> MCPServerInfo:
    """将服务器对象转换为MCPServerInfo响应对象"""
    return MCPServerInfo(
        id=server.server_config.id,
        description=server.server_config.description,
        connection_type=server.server_config.connection_type,
        command=server.server_config.command,
        args=" ".join(server.server_config.args) if isinstance(
            server.server_config.args, list) else server.server_config.args,
        url=getattr(server.server_config, 'url', None),
        connection_state=server.state.name.lower()
    )


@mcp_bp.route("/servers", methods=["GET"])
@require_auth
async def list_servers():
    """获取所有MCP服务器列表"""
    try:
        # 获取查询参数
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 20, type=int)
        connection_type = request.args.get('connection_type')
        status = request.args.get('status')
        query = request.args.get('query')

        # 从容器中获取MCP服务器管理器
        manager: MCPServerManager = g.container.resolve(MCPServerManager)

        # 获取所有服务器
        servers = manager.get_all_servers()

        # 转换为响应格式
        server_list = []
        for server_id, server in servers.items():
            # 过滤条件
            if connection_type and server.server_config.connection_type != connection_type:
                continue

            server_state = server.state.name.lower()
            if status:
                if status == 'connected' and server_state != 'connected':
                    continue
                elif status == 'disconnected' and server_state != 'disconnected':
                    continue
                elif status == 'error' and server_state != 'error':
                    continue

            if query and query.lower() not in server_id.lower() and (
                    not server.server_config.command or query.lower() not in server.server_config.command.lower()):
                continue

            server_list.append(_convert_to_server_info(server))

        # 计算分页
        total = len(server_list)
        total_pages = (total + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total)
        paginated_servers = server_list[start_idx:end_idx]

        # 返回响应
        return MCPServerList(
            items=[server for server in paginated_servers],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        ).model_dump()
    except Exception as e:
        logger.opt(exception=e).error("获取MCP服务器列表失败")
        return jsonify({"message": str(e)}), 500


@mcp_bp.route("/statistics", methods=["GET"])
@require_auth
async def get_statistics():
    """获取MCP服务器统计信息"""
    try:
        # 从容器中获取MCP服务器管理器
        manager: MCPServerManager = g.container.resolve(MCPServerManager)

        # 获取统计信息
        stats = manager.get_statistics()
        
        # 获取工具总数
        tools = manager.get_tools()
        total_tools = len(tools)

        # 返回响应
        return MCPStatistics(
            total_servers=stats.get("total", 0),
            stdio_servers=stats.get("stdio", 0),
            sse_servers=stats.get("sse", 0),
            connected_servers=stats.get("connected", 0),
            disconnected_servers=stats.get("disconnected", 0),
            error_servers=stats.get("error", 0),
            total_tools=total_tools
        ).model_dump()
    except Exception as e:
        logger.opt(exception=e).error("获取MCP统计信息失败")
        return jsonify({"message": str(e)}), 500


@mcp_bp.route("/servers/<server_id>", methods=["GET"])
@require_auth
async def get_server(server_id: str):
    """获取特定MCP服务器的详情"""
    try:
        # 从容器中获取MCP服务器管理器
        manager: MCPServerManager = g.container.resolve(MCPServerManager)

        # 获取服务器
        server = manager.get_server(server_id)
        if not server:
            return jsonify({"message": f"服务器 {server_id} 不存在"}), 404

        # 转换为响应格式
        server_info = _convert_to_server_info(server)

        # 返回响应
        return server_info.model_dump()
    except Exception as e:
        logger.opt(exception=e).error(f"获取MCP服务器 {server_id} 详情失败")
        return jsonify({"message": str(e)}), 500


@mcp_bp.route("/servers/<server_id>/tools", methods=["GET"])
@require_auth
async def get_server_tools(server_id: str):
    """获取MCP服务器提供的工具列表"""
    try:
        # 从容器中获取MCP服务器管理器
        manager: MCPServerManager = g.container.resolve(MCPServerManager)

        # 获取服务器
        server = manager.get_server(server_id)
        if not server:
            return jsonify({"message": f"服务器 {server_id} 不存在"}), 404

        # 如果服务器未连接，返回空列表
        if server.state != MCPConnectionState.CONNECTED:
            return []

        # 获取服务器工具
        tools = manager.get_tools()

        # 转换为响应格式
        tool_list = []
        for _, tool in tools.items():
            if tool.server_id == server_id:
                tool_list.append(MCPToolInfo(
                    name=tool.original_name,
                    description=tool.tool_info.description,
                    input_schema=tool.tool_info.inputSchema
                ))

        # 返回响应
        return [tool.model_dump() for tool in tool_list]
    except Exception as e:
        logger.opt(exception=e).error(f"获取MCP服务器 {server_id} 工具列表失败")
        return jsonify({"message": str(e)}), 500


@mcp_bp.route("/servers/check/<server_id>", methods=["GET"])
@require_auth
async def check_server_id(server_id: str):
    """检查服务器ID是否可用"""
    try:
        # 从容器中获取MCP服务器管理器
        manager: MCPServerManager = g.container.resolve(MCPServerManager)

        # 检查ID是否可用
        is_available = manager.is_server_id_available(server_id)

        # 返回响应
        return jsonify({
            "is_available": is_available
        })
    except Exception as e:
        logger.opt(exception=e).error(f"检查服务器ID {server_id} 可用性失败")
        return jsonify({"message": str(e)}), 500


@mcp_bp.route("/servers", methods=["POST"])
@require_auth
async def create_server():
    """创建新的MCP服务器"""
    try:
        # 获取请求数据
        data = await request.get_json()
        request_data = MCPServerCreateRequest(**data)

        # 从容器中获取全局配置和MCP服务器管理器
        config: GlobalConfig = g.container.resolve(GlobalConfig)
        manager: MCPServerManager = g.container.resolve(MCPServerManager)

        # 检查ID是否已存在
        if not manager.is_server_id_available(request_data.id):
            return jsonify({"message": f"服务器ID '{request_data.id}' 已存在或服务器正在运行"}), 409

        # 创建新的MCP服务器配置
        new_server_config = MCPServerConfig(
            id=request_data.id,
            description=request_data.description or "",
            command=request_data.command,
            args=request_data.args.split(" "),
            connection_type=request_data.connection_type,
            enable=True
        )

        # 添加到全局配置中
        config.mcp.servers.append(new_server_config)

        # 保存配置
        ConfigLoader.save_config_with_backup(CONFIG_FILE, config)

        # 让管理器加载新服务器
        server = manager.load_server(new_server_config)

        # 转换为响应格式
        server_info = _convert_to_server_info(server)

        # 返回响应
        return server_info.model_dump()
    except Exception as e:
        logger.opt(exception=e).error("创建MCP服务器失败")
        return jsonify({"message": str(e)}), 500


@mcp_bp.route("/servers/<server_id>", methods=["PUT"])
@require_auth
async def update_server(server_id: str):
    """更新MCP服务器配置"""
    try:
        # 获取请求数据
        data = await request.get_json()
        request_data = MCPServerUpdateRequest(**data)

        # 从容器中获取全局配置和MCP服务器管理器
        config: GlobalConfig = g.container.resolve(GlobalConfig)
        manager: MCPServerManager = g.container.resolve(MCPServerManager)

        # 查找服务器配置
        server_index = -1
        for i, server in enumerate(config.mcp.servers):
            if server.id == server_id:
                server_index = i
                break

        if server_index == -1:
            return jsonify({"message": f"服务器 '{server_id}' 不存在"}), 404

        # 检查服务器状态
        current_server = manager.get_server(server_id)
        if current_server and current_server.state == MCPConnectionState.CONNECTED:
            return jsonify({"message": "无法更新正在运行的服务器，请先停止服务器"}), 409

        # 更新服务器配置
        server_config = config.mcp.servers[server_index]

        if request_data.description is not None:
            server_config.description = request_data.description

        if request_data.command is not None:
            server_config.command = request_data.command

        if request_data.args is not None:
            server_config.args = request_data.args.split(" ")

        if request_data.connection_type is not None:
            server_config.connection_type = request_data.connection_type

        # 保存配置
        ConfigLoader.save_config_with_backup(CONFIG_FILE, config)

        # 停止服务器
        await manager.stop_server(server_id)

        # 重新加载服务器
        current_server = manager.load_server(server_config)

        try:
            await manager.connect_server(server_id)
        except Exception as e:
            logger.opt(exception=e).error(f"重新连接MCP服务器 {server_id} 失败")
            return jsonify({"message": str(e)}), 500

        # 转换为响应格式
        server_info = _convert_to_server_info(current_server)

        # 返回响应
        return server_info.model_dump()
    except Exception as e:
        logger.opt(exception=e).error(f"更新MCP服务器 {server_id} 失败")
        return jsonify({"message": str(e)}), 500


@mcp_bp.route("/servers/<server_id>", methods=["DELETE"])
@require_auth
async def delete_server(server_id: str):
    """删除MCP服务器"""
    try:
        # 从容器中获取全局配置和MCP服务器管理器
        config: GlobalConfig = g.container.resolve(GlobalConfig)
        manager: MCPServerManager = g.container.resolve(MCPServerManager)

        # 查找服务器配置
        server_index = -1
        for i, server in enumerate(config.mcp.servers):
            if server.id == server_id:
                server_index = i
                break

        if server_index == -1:
            return jsonify({"message": f"服务器 '{server_id}' 不存在"}), 404

        # 如果服务器正在运行，先停止它
        current_server = manager.get_server(server_id)
        if current_server and current_server.state == MCPConnectionState.CONNECTED:
            await manager.stop_server(server_id)

        # 从配置中删除服务器
        removed_server = config.mcp.servers.pop(server_index)

        # 保存配置
        ConfigLoader.save_config_with_backup(CONFIG_FILE, config)

        await manager.stop_server(server_id)

        # 返回响应
        return jsonify({})
    except Exception as e:
        logger.opt(exception=e).error(f"删除MCP服务器 {server_id} 失败")
        return jsonify({"message": str(e)}), 500


@mcp_bp.route("/servers/<server_id>/start", methods=["POST"])
@require_auth
async def start_server(server_id: str):
    """连接 MCP 服务器"""
    try:
        # 从容器中获取MCP服务器管理器
        manager: MCPServerManager = g.container.resolve(MCPServerManager)

        # 尝试连接服务器
        success = await manager.connect_server(server_id)

        if not success:
            return jsonify({"message": f"服务器 '{server_id}' 不存在或无法连接"}), 404

        # 返回响应
        return jsonify({})
    except Exception as e:
        logger.opt(exception=e).error(f"连接 MCP 服务器 {server_id} 失败")
        return jsonify({"message": str(e)}), 500


@mcp_bp.route("/servers/<server_id>/stop", methods=["POST"])
@require_auth
async def stop_server(server_id: str):
    """断开 MCP 服务器"""
    try:
        # 从容器中获取MCP服务器管理器
        manager: MCPServerManager = g.container.resolve(MCPServerManager)

        # 尝试停止服务器
        success = await manager.stop_server(server_id)

        if not success:
            return jsonify({"message": f"服务器 '{server_id}' 不存在或未连接"}), 404

        # 返回响应
        return jsonify({})
    except Exception as e:
        logger.opt(exception=e).error(f"断开 MCP 服务器 {server_id} 失败")
        return jsonify({"message": str(e)}), 500


@mcp_bp.route("/tools", methods=["GET"])
@require_auth
async def get_all_tools():
    """获取所有可用工具"""
    try:
        # 从容器中获取MCP服务器管理器
        manager: MCPServerManager = g.container.resolve(MCPServerManager)

        # 获取所有工具
        tools = manager.get_tools()

        # 转换为响应格式
        tool_list = []
        for name, tool_info in tools.items():
            tool_list.append(MCPToolInfo(
                name=name,
                description=tool_info.get("description", ""),
                input_schema=tool_info.get("input_schema", {})
            ))

        # 返回响应
        return [tool.model_dump() for tool in tool_list]
    except Exception as e:
        logger.opt(exception=e).error("获取所有工具失败")
        return jsonify({"message": str(e)}), 500
