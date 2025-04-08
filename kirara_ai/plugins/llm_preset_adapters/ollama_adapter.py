import asyncio
from typing import List, Optional, cast

import aiohttp
import requests
from pydantic import BaseModel, ConfigDict
from mcp.types import TextContent, ImageContent, EmbeddedResource

from kirara_ai.llm.adapter import AutoDetectModelsProtocol, LLMBackendAdapter
from kirara_ai.llm.format.message import (LLMChatContentPartType, LLMChatImageContent, LLMChatMessage,
                                          LLMChatTextContent, LLMToolCallContent, LLMToolResultContent)
from kirara_ai.llm.format.request import LLMChatRequest, Tool
from kirara_ai.llm.format.response import Function, LLMChatResponse, Message, ToolCall, Usage
from kirara_ai.logger import get_logger
from kirara_ai.media.manager import MediaManager
from kirara_ai.tracing import trace_llm_chat


class OllamaConfig(BaseModel):
    api_base: str = "http://localhost:11434"
    model_config = ConfigDict(frozen=True)

async def resolve_media_ids(media_ids: list[str], media_manager: MediaManager) -> List[str]:
    result = []
    for media_id in media_ids:
        media = media_manager.get_media(media_id)
        if media is not None:
            base64_data = await media.get_base64()
            result.append(base64_data)
    return result

def convert_llm_response(response_data: dict[str, dict]) -> list[LLMChatContentPartType]:
    # 通过实践证明 llm 调用工具时 content 字段为空字符串没有任何有效信息不进行记录
    if calls := response_data["message"].get("tool_calls", None):
        return [LLMToolCallContent(name=call["function"]["name"], parameters=call["function"].get("arguments", None)) for call in calls]
    else:
        return [LLMChatTextContent(text=response_data["message"].get("content", ""))]

def convert_non_tool_message(msg: LLMChatMessage, media_manager: MediaManager, loop: asyncio.AbstractEventLoop):
    text_content = ""
    images: list[str] = []
    for part in msg.content:
        if isinstance(part, LLMChatTextContent):
            text_content += part.text
        elif isinstance(part, LLMChatImageContent):
            images.append(part.media_id)
        elif isinstance(part, LLMToolCallContent):
            # 不太确定是否 ollama 需要tool_call信息。等待后续手动验证
            continue
    message = {"role": msg.role, "content": text_content}
    if images:
        message["images"] = loop.run_until_complete(resolve_media_ids(images, media_manager))
    return message


def resolve_tool_calls(response_data: dict[str, dict]) -> Optional[list[ToolCall]]:
    if tool_calls := response_data["message"].get("tool_calls", None):
        calls: list[ToolCall] = []
        for call in tool_calls:
            calls.append(ToolCall(
                function = Function(
                    name = call["function"]["name"], 
                    arguments = call["function"].get("arguments", None),
                )
            ))
        return calls
    else:
        return None

def convert_tools_to_ollama_format(tools: list[Tool]) -> list[dict]:
    # 这里将其独立出来方便应对后续接口改动
    return [tool.model_dump(exclude={"strict": True, "parameters": {"additionalProperties": True}}) for tool in tools]

def resolve_tool_results(element: LLMToolResultContent):
    if element.isError:
        return {"role": "tool", "content": f"An error occurred when calling the tool: {element.content}"}
    
    contents = []
    for content in element.content:
        if isinstance(content, TextContent):
            contents.append(content.text)
        elif isinstance(content, ImageContent):
            # 暂未支持
            continue
        elif isinstance(content, EmbeddedResource):
            # 暂未支持
            continue
    return {"role": "tool", "content": contents}

class OllamaAdapter(LLMBackendAdapter, AutoDetectModelsProtocol):
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.logger = get_logger("OllamaAdapter")
        
    @trace_llm_chat
    def chat(self, req: LLMChatRequest) -> LLMChatResponse:
        api_url = f"{self.config.api_base}/api/chat"
        headers = {"Content-Type": "application/json"}

        # 将消息转换为 Ollama 格式
        messages = []
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        for msg in req.messages:
            # 收集每条消息中的文本内容和图像
            if msg.role == "tool":
                # 官网没有如何传递 tool_result 的例子，这是查看多篇教程后得出的结论
                # 目前 ollama 不需要 tool_call 信息，判断结果是否需要估计根据上下文推断。Tips: 顺序至关重要
                parts = cast(list[LLMToolResultContent], msg.content)
                messages.extend([{"role": "tool", "content": part.content} for part in parts])
            else:
                messages.append(convert_non_tool_message(msg, self.media_manager, loop))
                
        data = {
            "model": req.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": req.temperature,
                "top_p": req.top_p,
                "num_predict": req.max_tokens,
                "stop": req.stop,
            "tools": convert_tools_to_ollama_format(req.tools) if req.tools else None,
            },
        }

        # Remove None fields
        data = {k: v for k, v in data.items() if v is not None}
        if "options" in data:
            data["options"] = {
                k: v for k, v in data["options"].items() if v is not None  # type: ignore
            }

        response = requests.post(api_url, json=data, headers=headers)
        try:
            response.raise_for_status()
            response_data = response.json()
        except Exception as e:
            self.logger.error(f"API Response: {response.text}")
            raise e
        # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion

        return LLMChatResponse(
            model=req.model,
            message=Message(
                content= convert_llm_response(response_data),
                role="assistant",
                finish_reason="stop",
                tool_calls= resolve_tool_calls(response_data),
            ),
            usage=Usage(
                prompt_tokens=response_data['prompt_eval_count'],
                completion_tokens=response_data['eval_count'],
                total_tokens=response_data['prompt_eval_count'] + response_data['eval_count'],
            )
        )

    async def auto_detect_models(self) -> list[str]:
        api_url = f"{self.config.api_base}/api/tags"
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(api_url) as response:
                response.raise_for_status()
                response_data = await response.json()
                return [tag["name"] for tag in response_data["models"]]