import asyncio
from typing import Any, Dict, List, Optional

from mcp import types

from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.llm.format.message import LLMChatMessage, LLMChatTextContent, LLMToolResultContent
from kirara_ai.llm.format.request import LLMChatRequest
from kirara_ai.llm.format.response import LLMChatResponse
from kirara_ai.logger import get_logger
from kirara_ai.mcp.manager import MCPServerManager
from kirara_ai.workflow.core.block import Block, Input, Output


class MCPCallTool(Block):
    """
    MCP工具调用模块，用于调用MCP服务器提供的工具
    
    此模块接收来自FunctionCalling的工具调用请求，通过MCPServerManager调用相应的工具，
    并将调用结果格式化后返回，以便继续与LLM交互。
    """
    name = "mcp_call_tool"
    inputs = {
        "raw_input": Input("raw_input", "原始输入", List[LLMChatMessage], "原始输入", False),
        "tool_call": Input("tool_call", "工具调用请求", LLMChatResponse, "LLM返回的工具调用请求", False),
    }
    outputs = {
        "updated_request": Output("updated_request", "更新后的请求体", List[LLMChatMessage], "包含工具调用结果的完整请求体")
    }
    container: DependencyContainer

    def __init__(self):
        self.logger = get_logger("MCPCallTool")

    def execute(self, raw_input: List[LLMChatMessage], tool_call: LLMChatResponse) -> Dict[str, Any]:
        """
        执行MCP工具调用
        
        Args:
            im_msg: 用户发送的IM消息
            tool_call: LLM返回的工具调用请求
            
        Returns:
            包含记忆内容和请求体的字典
        """
        if not tool_call or not tool_call.message.tool_calls:
            self.logger.error("没有收到有效的工具调用请求")
            return self._create_error_response("没有收到有效的工具调用请求")
        
        # 获取MCP服务器管理器
        mcp_manager = self.container.resolve(MCPServerManager)
        if not mcp_manager:
            self.logger.error("无法获取MCP服务器管理器")
            return self._create_error_response("系统错误：无法获取MCP服务器管理器")
        
        # 获取事件循环
        loop = self.container.resolve(asyncio.AbstractEventLoop)
        
        # 处理所有工具调用
        tool_results: List[LLMToolResultContent] = []
        for tool_call_info in tool_call.message.tool_calls:
            try:
                assert tool_call_info.function is not None
                assert tool_call_info.function.name is not None
                assert tool_call_info.function.arguments is not None
                # 获取工具名称和参数
                tool_name = tool_call_info.function.name
                tool_args = tool_call_info.function.arguments
                tool_id = tool_call_info.id
                
                self.logger.info(f"调用工具: {tool_name}, 参数: {tool_args}")
                
                # 获取工具对应的服务器
                server_info = mcp_manager.get_tool_server(tool_name)
                if not server_info:
                    error_msg = f"找不到工具: {tool_name}"
                    self.logger.error(error_msg)
                    tool_results.extend(self._create_tool_result(tool_id, tool_name, {"error": error_msg}))
                    continue
                
                server, original_name = server_info
                
                # 调用工具
                result_future = asyncio.run_coroutine_threadsafe(server.call_tool(original_name, tool_args), loop)
                result = result_future.result()
                
                # 创建工具结果
                tool_results.extend(self._create_tool_result(tool_id, tool_name, result.content))
                self.logger.info(f"工具调用结果: {tool_results[-1]}")
            except Exception as e:
                self.logger.opt(exception=e).error(f"调用工具时发生错误")
                error_msg = f"调用工具时发生错误: {str(e)}"
                tool_results.extend(self._create_tool_result(
                    tool_id if 'tool_id' in locals() else None, 
                    tool_name if 'tool_name' in locals() else "unknown", 
                    {"error": error_msg}
                ))
        
        # 创建工具结果消息
        tool_result_message = LLMChatMessage(
            role="tool",
            content=tool_results
        )
    
        
        return {
            "updated_request": raw_input + [tool_result_message]
        }
    
    def _create_tool_result(self, tool_id: Optional[str], tool_name: str, content: list[types.TextContent | types.ImageContent | types.EmbeddedResource]) -> list[LLMToolResultContent]:
        """创建工具调用结果"""
        result_content = []
        
        for item in content:
            if isinstance(item, types.TextContent):
                result_content.append(LLMToolResultContent(
                    id=tool_id,
                    name=tool_name,
                    content=item.text
                ))
            elif isinstance(item, types.ImageContent):
                result_content.append(LLMToolResultContent(
                    id=tool_id,
                    name=tool_name,
                    content=item.image_url
                ))
            elif isinstance(item, types.EmbeddedResource):
                result_content.append(LLMToolResultContent(
                    id=tool_id,
                    name=tool_name,
                    content=item.resource_url
                ))
            else:
                result_content.append(LLMToolResultContent(
                    id=tool_id,
                    name=tool_name,
                    content=str(item)
                ))
        
        return result_content

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """创建错误响应"""
        error_text = LLMChatTextContent(text=error_message)
        assistant_message = LLMChatMessage(role="assistant", content=[error_text])
        
        return {
            "send_memory": [assistant_message],
            "request_body": LLMChatRequest(messages=[assistant_message])
        }
