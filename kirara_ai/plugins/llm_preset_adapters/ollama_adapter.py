import asyncio
from typing import Any, List, cast

import aiohttp
import requests
from pydantic import BaseModel, ConfigDict
from mcp.types import TextContent, ImageContent, EmbeddedResource

import kirara_ai.llm.format.tool as tools
from kirara_ai.llm.adapter import AutoDetectModelsProtocol, LLMBackendAdapter
from kirara_ai.llm.format.message import (LLMChatContentPartType, LLMChatImageContent, LLMChatMessage,
                                          LLMChatTextContent, LLMToolCallContent, LLMToolResultContent)
from kirara_ai.llm.format.request import LLMChatRequest, Tool
from kirara_ai.llm.format.response import LLMChatResponse, Message, Usage
from kirara_ai.logger import get_logger
from kirara_ai.media.manager import MediaManager
from kirara_ai.tracing import trace_llm_chat

from .openai_adapter import convert_tools_to_openai_format
from .utils import generate_tool_call_id, pick_tool_calls


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

def convert_llm_response(response_data: dict[str, dict[str, Any]]) -> list[LLMChatContentPartType]:
    # 通过实践证明 llm 调用工具时 content 字段为空字符串没有任何有效信息不进行记录
    if calls := response_data["message"].get("tool_calls", None):
        return [LLMToolCallContent(
            id=generate_tool_call_id(call["function"]["name"]),
            name=call["function"]["name"],
            parameters=call["function"].get("arguments", None)
        ) for call in calls
        ]
    else:
        return [LLMChatTextContent(text=response_data["message"].get("content", ""))]

def convert_non_tool_message(msg: LLMChatMessage, media_manager: MediaManager, loop: asyncio.AbstractEventLoop) -> dict[str, Any]:
    text_content = ""
    images: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    messages: dict[str, Any] = {
        "role": msg.role,
        "content": "",
    }
    for part in msg.content:
        if isinstance(part, LLMChatTextContent):
            text_content += part.text
        elif isinstance(part, LLMChatImageContent):
            images.append(part.media_id)
        elif isinstance(part, LLMToolCallContent):
            tool_calls.append({
                "function": {
                    "name": part.name,
                    "arguments": part.parameters,
                }
            })
    messages["content"] = text_content
    if images:
        messages["images"] = loop.run_until_complete(
            resolve_media_ids(images, media_manager))
    if tool_calls:
        messages["tool_calls"] = tool_calls
    return messages


def convert_tool_result_message(msg: LLMChatMessage, media_manager: MediaManager, loop: asyncio.AbstractEventLoop) -> list[dict]:
    """
    将工具调用结果转换为 Ollama 格式
    """
    elements = cast(list[LLMToolResultContent], msg.content)
    messages = []
    for element in elements:
        output = ""
        for item in element.content:
            if isinstance(item, tools.TextContent):
                output += f"{item.text}\n"
            elif isinstance(item, tools.MediaContent):
                output += f"<media id={item.media_id} mime_type={item.mime_type} />\n"
        if element.isError:
            output = f"Error: {element.name}\n{output}"
        messages.append({"role": "tool", "content": output,
                        "tool_call_id": element.id})
    return messages

def convert_tools_to_ollama_format(tools: list[Tool]) -> list[dict]:
    # 这里将其独立出来方便应对后续接口改动
    return convert_tools_to_openai_format(tools)

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
                messages.extend(convert_tool_result_message(
                    msg, self.media_manager, loop))
            else:
                messages.append(convert_non_tool_message(
                    msg, self.media_manager, loop))

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
                k: v for k, v in data["options"].items() if v is not None # type: ignore
            }

        response = requests.post(api_url, json=data, headers=headers)
        try:
            response.raise_for_status()
            response_data = response.json()
        except Exception as e:
            self.logger.error(f"API Response: {response.text}")
            raise e
        # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
        content = convert_llm_response(response_data)
        return LLMChatResponse(
            model=req.model,
            message=Message(
                content=content,
                role="assistant",
                finish_reason="stop",
                tool_calls=pick_tool_calls(content),
            ),
            usage=Usage(
                prompt_tokens=response_data['prompt_eval_count'],
                completion_tokens=response_data['eval_count'],
                total_tokens=response_data['prompt_eval_count'] +
                response_data['eval_count'],
            )
        )

    async def auto_detect_models(self) -> list[str]:
        api_url = f"{self.config.api_base}/api/tags"
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(api_url) as response:
                response.raise_for_status()
                response_data = await response.json()
                return [tag["name"] for tag in response_data["models"]]
