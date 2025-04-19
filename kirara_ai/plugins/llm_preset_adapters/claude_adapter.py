import asyncio
import base64
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from pydantic import BaseModel, ConfigDict

import kirara_ai.llm.format.tool as tools
from kirara_ai.llm.adapter import AutoDetectModelsProtocol, LLMBackendAdapter
from kirara_ai.llm.format.message import (LLMChatContentPartType, LLMChatImageContent, LLMChatMessage,
                                          LLMChatTextContent, LLMToolCallContent, LLMToolResultContent)
from kirara_ai.llm.format.request import LLMChatRequest, Tool
from kirara_ai.llm.format.response import LLMChatResponse, Message, ToolCall, Usage
from kirara_ai.llm.format.tool import Function, ToolCall
from kirara_ai.logger import get_logger
from kirara_ai.media.manager import MediaManager
from kirara_ai.tracing.decorator import trace_llm_chat


class ClaudeConfig(BaseModel):
    api_key: str
    api_base: str = "https://api.anthropic.com/v1"
    model_config = ConfigDict(frozen=True)


async def convert_llm_chat_message_to_claude_message(messages: list[LLMChatMessage], media_manager: MediaManager) -> list[dict]:
    content: List[Dict[str, Any]] = []
    for msg in [msg for msg in messages if msg.role in ["user", "assistant", "tool"]]:
        parts: List[Dict[str, Any]] = []
        for part in msg.content:
            if isinstance(part, LLMChatTextContent):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, LLMToolResultContent):
                tool_result: List[Dict[str, Any]] = []
                for item in part.content:
                    if isinstance(item, tools.TextContent):
                        tool_result.append({"type": "text", "text": item.text})
                    elif isinstance(item, tools.MediaContent):
                        media = media_manager.get_media(item.media_id)
                        if media is None:
                            raise ValueError(
                                f"Media {item.media_id} not found")
                        tool_result.append({
                            "type": media.media_type.value.lower(),
                            "source": {
                                "type": "base64", "media_type": str(media.mime_type), "data": await media.get_base64()
                            }
                        })
                parts.append({
                    "type": "tool_result",
                    "tool_use_id": part.id,
                    "content": tool_result,
                    "is_error": part.isError
                })
            elif isinstance(part, LLMToolCallContent):
                # claude不需要tool_call内容
                continue
            elif isinstance(part, LLMChatImageContent):
                media = media_manager.get_media(part.media_id)
                if media is None:
                    raise ValueError(f"Media {part.media_id} not found")
                parts.append({"source": {"media_type": str(media.mime_type), "data": await media.get_base64()}, "type": "image"})
        content.append({
            "role": "user" if msg.role == "tool" else msg.role,
            "content": parts
        })
    return content


def resolve_tool_calls(content: list[dict]) -> Optional[list[ToolCall]]:
    tool_calls = []
    for part in content:
        if part.get("type") == "tool_use":
            tool_calls.append(
                ToolCall(
                    model="claude",
                    id=part.get("id"),
                    type=part.get("type"),
                    function=Function(
                        name=part.get("name", ""),
                        arguments=part.get("input", None),
                    )
                )
            )
    # 当tool_calls为空时，返回None
    if tool_calls:
        return tool_calls
    else:
        return None


def convert_tools_to_claude_format(tools: list[Tool]) -> list[dict]:
    # 使用 pydantic 的 model_dump 方法，高级排除项`exclude`排除 openai 专属项
    return [tool.model_dump(exclude={"strict": True, 'parameters': {'additionalProperties': True}}) for tool in tools]


class ClaudeAdapter(LLMBackendAdapter, AutoDetectModelsProtocol):

    media_manager: MediaManager

    def __init__(self, config: ClaudeConfig):
        self.config = config
        self.logger = get_logger("ClaudeAdapter")

    @trace_llm_chat
    def chat(self, req: LLMChatRequest) -> LLMChatResponse:
        api_url = f"{self.config.api_base}/messages"
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # Claude 的系统消息比较特殊
        system_messages = [msg for msg in req.messages if msg.role == "system"]
        if system_messages:
            system_message = system_messages[0].content
        else:
            system_message = None

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # 构建请求数据

        data = {
            "model": req.model,
            "messages": loop.run_until_complete(convert_llm_chat_message_to_claude_message(req.messages, self.media_manager)),
            "max_tokens": req.max_tokens,
            "system": system_message,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "stream": req.stream,
            # claude tools格式中参数部分命名与openai api不同，不能简单使用model_dumps，在这里进行转换
            "tools": convert_tools_to_claude_format(req.tools) if req.tools else None,
            # claude默认如果使用了tools字段，这里需要指定tool_choice， claude默认为{"type": "auto"}.
            # 可考虑后续给用户暴露此接口， 目前此处各模型定义不太统一
            "tool_choice": {"type": "auto"} if req.tools else None,
        }
        # Remove None fields
        data = {k: v for k, v in data.items() if v is not None}

        response = requests.post(api_url, json=data, headers=headers)
        try:
            response.raise_for_status()
            response_data = response.json()
        except Exception as e:
            self.logger.error(f"API Response: {response.text}")
            raise e

        content: List[LLMChatContentPartType] = []

        for res in response_data["content"]:
            if res["type"] == "text":
                content.append(LLMChatTextContent(text=res["text"]))
            elif res["type"] == "image":
                image_data = base64.b64decode(res["source"]["data"])
                media = loop.run_until_complete(self.media_manager.register_from_data(
                    image_data, res["source"]["media_type"], source="claude response"))
                content.append(LLMChatImageContent(media_id=media))
            elif res["type"] == "tool_use":
                # tool_call 时 只会额外返回一个 text 的深度思考。
                content.append(LLMToolCallContent(id=res.get("id", None), name=res["name"], parameters=res.get("input", None)))
        usage_data = response_data.get("usage", {})
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)

        return LLMChatResponse(
            model=req.model,
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            message=Message(
                content=content,
                role=response_data.get("role", "assistant"),
                finish_reason=response_data.get("stop_reason", "stop"),
                # claude tool_call混合在content字段中，需要提取
                tool_calls=resolve_tool_calls(response_data["content"]),
            )
        )

    async def auto_detect_models(self) -> list[str]:
        # {
        #   "data": [
        #     {
        #       "type": "model",
        #       "id": "claude-3-5-sonnet-20241022",
        #       "display_name": "Claude 3.5 Sonnet (New)",
        #       "created_at": "2024-10-22T00:00:00Z"
        #     }
        #   ],
        #   "has_more": true,
        #   "first_id": "<string>",
        #   "last_id": "<string>"
        # }
        # claude3 全系支持工具调用，支持多模态tool_result
        api_url = f"{self.config.api_base}/models"
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(
                api_url, headers={"x-api-key": self.config.api_key}
            ) as response:
                response.raise_for_status()
                response_data = await response.json()
                return [model["id"] for model in response_data["data"]]
