from .openai_adapter import (
    OpenAIAdapterChatBase,
    OpenAIConfig,
    convert_llm_chat_message_to_openai_message,
    convert_tools_to_openai_format,
    pick_tool_calls
)
from kirara_ai.llm.format.request import LLMChatRequest
from kirara_ai.llm.format.response import LLMChatResponse, Message, Usage
from kirara_ai.llm.format.message import (
    LLMChatContentPartType, 
    LLMChatTextContent,
    LLMToolCallContent
)
from kirara_ai.tracing import trace_llm_chat
from kirara_ai.logger import get_logger
import asyncio
import requests
import json
from typing import Any, Dict, List, cast, Optional

logger = get_logger("ModelScopeAdapter")

class ModelScopeConfig(OpenAIConfig):
    """ModelScope API 配置，继承自 OpenAI 配置但使用不同的 API 地址"""
    api_base: str = "https://api-inference.modelscope.cn/v1"

class ModelScopeAdapter(OpenAIAdapterChatBase):
    """
    ModelScope API 适配器实现
    """
    
    def __init__(self, config: ModelScopeConfig):
        super().__init__(config)
    
    @trace_llm_chat
    def chat(self, req: LLMChatRequest) -> LLMChatResponse:
        api_url = f"{self.config.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        
        # 构建请求数据
        model_name = req.model or ""
        data = {
            "messages": asyncio.run(convert_llm_chat_message_to_openai_message(req.messages, self.media_manager)),
            "model": model_name,
            "frequency_penalty": req.frequency_penalty,
            "max_completion_tokens": req.max_tokens,
            "presence_penalty": req.presence_penalty,
            "response_format": req.response_format,
            "stop": req.stop,
            "stream": req.stream,
            "stream_options": req.stream_options,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "tools": convert_tools_to_openai_format(req.tools) if req.tools else None,
            "tool_choice": "auto" if req.tools else None,
            "logprobs": req.logprobs,
            "top_logprobs": req.top_logprobs,
        }
        
        # ========== ModelScope 特有参数处理 ==========
        # 智能处理 enable_thinking 参数
        data["enable_thinking"] = False
        # data["stream"] = True
        # 移除 None 字段
        data = {k: v for k, v in data.items() if v is not None}
        
        logger.debug(f"Sending request to ModelScope API: {api_url}")
        logger.debug(f"Request data: {json.dumps(data, indent=2, ensure_ascii=False)}")
        if req.model not in  ["Qwen/QwQ-32B","deepseek-ai/DeepSeek-R1-0528"]:
            return self._handle_non_streaming_request(api_url, headers, data, model_name)
        else:
            return self._handle_streaming_request(api_url, headers, model_name, req)
    
    def _handle_streaming_request(self, api_url: str, headers: dict, model: str, req:LLMChatRequest) -> LLMChatResponse:
        """处理流式请求，当模型为Qwen/QwQ-32B时"""
        content_text = ""
        if req.messages and req.messages[0].content:
            first_content = req.messages[0].content[0]
            if isinstance(first_content, LLMChatTextContent):
                content_text = first_content.text
            else:
                # 处理非文本内容（如图像、工具调用等）
                content_text = f"[Non-text content: {type(first_content).__name__}]"
        
        # 显式声明 messages 为列表类型
        messages: List[Dict[str, str]] = [{
            "role": "user",
            "content": content_text
        }]
        
        # 显式声明 data 类型
        data: Dict[str, Any] = {
            "stream": True,
            "messages": messages,
            "model": model,
        }
        response = requests.post(api_url, json=data, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # 处理流式响应
        full_content = ""
        tool_calls = []
        finish_reason: Optional[str] = None
        role: Optional[str] = None
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        for line in response.iter_lines():
            if not line:
                continue
                
            # 解析SSE事件
            if line.startswith(b"data: "):
                event_data = line[6:].decode('utf-8')
                if event_data == "[DONE]":
                    break
                    
                try:
                    chunk = json.loads(event_data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON: {event_data}")
                    continue
                    
                # 处理内容块
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    
                    # 只收集最终答案内容
                    if "content" in delta and delta["content"]:
                        full_content += delta["content"]
                    
                    # 记录角色信息
                    if "role" in delta and delta["role"]:
                        role = delta["role"]
                    
                    # 处理工具调用（简化版）
                    if "tool_calls" in delta and delta["tool_calls"]:
                        tool_calls = delta["tool_calls"]  # 直接覆盖，不合并片段
                    
                    # 记录完成原因
                    finish_reason = choices[0].get("finish_reason", finish_reason)
                
                # 收集使用量统计
                if "usage" in chunk:
                    usage = chunk["usage"]
        
        # 构建最终响应内容
        content_parts: List[LLMChatContentPartType] = []
        
        if tool_calls:
            # 简化工具调用处理
            content_parts = [
                LLMToolCallContent(
                    id=call["id"],
                    name=call["function"]["name"],
                    parameters=json.loads(call["function"].get("arguments", "{}"))
                )
                for call in tool_calls
            ]
        else:
            # 只返回最终内容
            content_parts = [LLMChatTextContent(text=full_content)]

        # 安全获取角色值
        role_value = role or "assistant"
        
        return LLMChatResponse(
            model=model,
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            message=Message(
                content=content_parts,
                role=role_value,
                tool_calls=pick_tool_calls(content_parts),
                finish_reason=finish_reason or "",
            ),
        )


    def _handle_non_streaming_request(self, api_url: str, headers: dict, data: dict, model: str) -> LLMChatResponse:
        """处理非流式请求"""
        response = requests.post(api_url, json=data, headers=headers, timeout=15)
        
        try:
            response.raise_for_status()
            response_data: dict = response.json()
        except Exception as e:
            logger.error(f"ModelScope API error: {response.text}")
            raise
        
        logger.debug(f"Received non-streaming response: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
        
        return self._parse_response(response_data, model)
    
    def _parse_response(self, response_data: dict, model: str) -> LLMChatResponse:
        """解析非流式响应"""
        choices: List[dict[str, Any]] = response_data.get("choices", [{}])
        first_choice = choices[0] if choices else {}
        message: dict[str, Any] = first_choice.get("message", {})
        
        # 处理工具调用或普通响应
        content: list[LLMChatContentPartType] = []
        if tool_calls := message.get("tool_calls", None):
            content = [
                LLMToolCallContent(
                    id=call["id"],
                    name=call["function"]["name"],
                    parameters=json.loads(call["function"].get("arguments", "{}"))
                ) for call in tool_calls
            ]
        else:
            # 普通文本响应
            content = [LLMChatTextContent(text=message.get("content", ""))]
        
        # 处理使用统计
        usage_data = response_data.get("usage", {})
        
        return LLMChatResponse(
            model=model,
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            message=Message(
                content=content,
                role=message.get("role", "assistant"),
                tool_calls=pick_tool_calls(content),
                finish_reason=first_choice.get("finish_reason", ""),
            ),
        )
       