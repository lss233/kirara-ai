import asyncio
import json
from typing import Any, Dict, List, cast

import aiohttp
import requests
from pydantic import BaseModel, ConfigDict

import kirara_ai.llm.format.tool as tools
from kirara_ai.llm.adapter import AutoDetectModelsProtocol, LLMBackendAdapter
from kirara_ai.llm.format.message import (LLMChatContentPartType, LLMChatImageContent, LLMChatMessage,
                                          LLMChatTextContent, LLMToolCallContent, LLMToolResultContent)
from kirara_ai.llm.format.request import LLMChatRequest, Tool
from kirara_ai.llm.format.response import LLMChatResponse, Message, Usage
from kirara_ai.logger import get_logger
from kirara_ai.media import MediaManager
from kirara_ai.tracing import trace_llm_chat

from .utils import pick_tool_calls

logger = get_logger("OpenAIAdapter")

async def convert_parts_factory(messages: LLMChatMessage, media_manager: MediaManager) -> list[dict]:
    if messages.role == "tool":
        # typing.cast 指定类型，避免mypy报错
        elements = cast(list[LLMToolResultContent], messages.content)
        outputs = []
        for element in elements:
            # 保证 content 为一个字符串
            output = ""
            for content in element.content:
                if isinstance(content, tools.TextContent):
                    output = content.text
                elif isinstance(content, tools.MediaContent):
                    media = media_manager.get_media(content.media_id)
                    if media is None:
                        raise ValueError(f"Media {content.media_id} not found")
                    output += f"<media id={content.media_id} mime_type={content.mime_type} />"
                else:
                    raise ValueError(f"Unsupported content type: {type(content)}")
            if element.isError:
                output = f"Error: {element.name}\n" + output
            outputs.append({
                "role": "tool",
                "tool_call_id": element.id,
                "content": output,
            })
        return outputs
    else:
        parts = []
        elements = cast(list[LLMChatContentPartType], messages.content)
        tool_calls: list[dict[str, Any]] = []
        for element in elements:
            if isinstance(element, LLMChatTextContent):
                parts.append(element.model_dump(mode="json"))
            elif isinstance(element, LLMChatImageContent):
                media = media_manager.get_media(element.media_id)
                if media is None:
                    raise ValueError(f"Media {element.media_id} not found")
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": await media.get_base64_url()
                    }
                })
            elif isinstance(element, LLMToolCallContent):
                tool_calls.append({
                    "type": "function",
                    "id": element.id,
                    "function": {
                        "name": element.name,
                        "arguments": json.dumps(element.parameters or {}, ensure_ascii=False),
                    }
                })
        response: Dict[str, Any] = {"role": messages.role}
        if parts:
            response["content"] = parts
        if tool_calls:
            response["tool_calls"] = tool_calls
        return [response]

def convert_llm_chat_message_to_openai_message(messages: list[LLMChatMessage], media_manager: MediaManager, loop: asyncio.AbstractEventLoop) -> list[dict]:
    results = loop.run_until_complete(
        asyncio.gather(*[convert_parts_factory(msg, media_manager) for msg in messages])
    )
    # 扁平化结果, 展开所有列表
    return [item for sublist in results for item in sublist]

def convert_tools_to_openai_format(tools: list[Tool]) -> list[dict]:
    return [{
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters if isinstance(tool.parameters, dict) else tool.parameters.model_dump(),
            "strict": tool.strict,
        }
    } for tool in tools]

class OpenAIConfig(BaseModel):
    api_key: str
    api_base: str = "https://api.openai.com/v1"
    model_config = ConfigDict(frozen=True)


class OpenAIAdapter(LLMBackendAdapter, AutoDetectModelsProtocol):
    media_manager: MediaManager
    
    def __init__(self, config: OpenAIConfig):
        self.config = config
    @trace_llm_chat
    def chat(self, req: LLMChatRequest) -> LLMChatResponse:
        api_url = f"{self.config.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        data = {
            "messages": convert_llm_chat_message_to_openai_message(req.messages, self.media_manager, loop),
            "model": req.model,
            "frequency_penalty": req.frequency_penalty,
            "max_tokens": req.max_tokens,
            "presence_penalty": req.presence_penalty,
            "response_format": req.response_format,
            "stop": req.stop,
            "stream": req.stream,
            "stream_options": req.stream_options,
            "temperature": req.temperature,
            "top_p": req.top_p,
            # tool pydantic 模型按照 openai api 格式进行的建立。所以这里直接dump
            "tools": convert_tools_to_openai_format(req.tools) if req.tools else None,
            "tool_choice": "auto" if req.tools else None,
            "logprobs": req.logprobs,
            "top_logprobs": req.top_logprobs,
        }

        # Remove None fields
        data = {k: v for k, v in data.items() if v is not None}
        
        logger.debug(f"Request: {data}")

        response = requests.post(api_url, json=data, headers=headers)
        try:
            response.raise_for_status()
            response_data: dict = response.json()
        except Exception as e:
            logger.error(f"Response: {response.text}")
            raise e
        logger.debug(f"Response: {response_data}")

        choices: List[dict[str, Any]] = response_data.get("choices", [{}])
        first_choice = choices[0] if choices else {}
        message: dict[str, Any] = first_choice.get("message", {})
        
        # 检测tool_calls字段是否存在和是否不为None. tool_call时content字段无有效信息，暂不记录
        content: list[LLMChatContentPartType] = []
        if tool_calls := message.get("tool_calls", None):
            content = [LLMToolCallContent(
                id=call["id"],
                name=call["function"]["name"],
                parameters=json.loads(call["function"].get("arguments", "{}"))
            ) for call in tool_calls]
        else:
            content = [LLMChatTextContent(text=message.get("content", ""))]

        usage_data = response_data.get("usage", {})
        
        return LLMChatResponse(
            model=req.model,
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            message=Message(
                content=content,
                role=message.get("role", "assistant"),
                tool_calls = pick_tool_calls(content),
                finish_reason=first_choice.get("finish_reason", ""),
            ),
        )

    async def auto_detect_models(self) -> list[str]:
        api_url = f"{self.config.api_base}/models"
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(
                api_url, headers={"Authorization": f"Bearer {self.config.api_key}"}
            ) as response:
                response.raise_for_status()
                response_data = await response.json()
                return [model["id"] for model in response_data["data"]]