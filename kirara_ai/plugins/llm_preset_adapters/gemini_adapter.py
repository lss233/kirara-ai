import asyncio
import base64
from typing import Any, Dict, List, Literal, Optional, cast

import aiohttp
import requests
from pydantic import BaseModel, ConfigDict

from kirara_ai.llm.adapter import AutoDetectModelsProtocol, LLMBackendAdapter
from kirara_ai.llm.format.message import (LLMChatContentPartType, LLMChatImageContent, LLMChatMessage,
                                          LLMChatTextContent, LLMToolCallContent, LLMToolResultContent)
from kirara_ai.llm.format.request import LLMChatRequest, Tool
from kirara_ai.llm.format.response import Function, LLMChatResponse, Message, ToolCall, Usage
from kirara_ai.logger import get_logger
from kirara_ai.media import MediaManager
from kirara_ai.tracing import trace_llm_chat

SAFETY_SETTINGS = [{
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
}, {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
}, {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
}, {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
}, {
    "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
    "threshold": "BLOCK_NONE"
}]

# POST 模式支持最大 20 MB 的 inline data
INLINE_LIMIT_SIZE = 1024 * 1024 * 20

IMAGE_MODAL_MODELS = [
    "gemini-2.0-flash-exp"
]


class GeminiConfig(BaseModel):
    api_key: str
    api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    model_config = ConfigDict(frozen=True)


async def convert_non_tool_message(msg: LLMChatMessage, media_manager: MediaManager) -> dict:
    parts: List[Dict[str, Any]] = []
    elements = cast(list[LLMChatContentPartType], msg.content)
    for element in elements:
        if isinstance(element, LLMChatTextContent):
            parts.append({"text": element.text})
        elif isinstance(element, LLMChatImageContent):
            media = media_manager.get_media(element.media_id)
            if media is None:
                raise ValueError(f"Media {element.media_id} not found")
            parts.append({
                "inline_data": {
                    "mime_type": str(media.mime_type),
                    "data": await media.get_base64()
                }
            })
        elif isinstance(element, LLMToolCallContent):
            parts.append({
                "functionCall": {
                    "name": element.name,
                    "args": element.parameters 
                }
            })
    return {
        "role": "model" if msg.role == "assistant" else "user",
        "parts": parts
    }

async def convert_llm_chat_message_to_gemini_message(msg: LLMChatMessage, media_manager: MediaManager) -> dict:
    if msg.role in ["user", "assistant", "system"]:
        return await convert_non_tool_message(msg, media_manager)
    elif msg.role == "tool":
        elements = cast(list[LLMToolResultContent], msg.content)
        # tool_result 的 role 必定为 "tool"。同时 "tool" 角色 messages 中只能是 LLMToolResultContent
        parts = [{
            "functionResponse": {
                "name": element.name,
                "response": {
                    "name": element.name,
                    "content": element.content
                }
            }
        } for element in elements]

        return {"parts": parts}

def resolve_function_call(calls: list[LLMChatContentPartType]) -> Optional[list[ToolCall]]:
    tool_calls = [
        ToolCall(
            model="gemini",
            function=Function(name=call.name, arguments=call.parameters)
        ) for call in calls if isinstance(call, LLMToolCallContent)
    ]
    if tool_calls:
        return tool_calls
    else:
        return None

def convert_tools_to_gemini_format(tools: list[Tool]) -> list[dict[Literal["function_declarations"], list[dict]]]:
    return [{
        "function_declarations": [tool.model_dump(exclude={"strict": True, "parameters": {"additionalProperties": True}}) for tool in tools]
    }]

class GeminiAdapter(LLMBackendAdapter, AutoDetectModelsProtocol):

    media_manager: MediaManager

    def __init__(self, config: GeminiConfig):
        self.config = config
        self.logger = get_logger("GeminiAdapter")

    @trace_llm_chat
    def chat(self, req: LLMChatRequest) -> LLMChatResponse:
        api_url = f"{self.config.api_base}/models/{req.model}:generateContent"
        headers = {
            "x-goog-api-key": self.config.api_key,
            "Content-Type": "application/json",
        }

        # create a new asyncio loop to run the convert_llm_chat_message_to_gemini_message function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        response_modalities = ["text"]
        if req.model in IMAGE_MODAL_MODELS:
            response_modalities.append("image")

        data = {
            "contents": loop.run_until_complete(asyncio.gather(*[convert_llm_chat_message_to_gemini_message(msg, self.media_manager) for msg in req.messages])),
            "generationConfig": {
                "temperature": req.temperature,
                "topP": req.top_p,
                "topK": 40,
                "maxOutputTokens": req.max_tokens,
                "stopSequences": req.stop,
                "responseModalities": response_modalities,
            },
            "safetySettings": SAFETY_SETTINGS,
            "tools": convert_tools_to_gemini_format(req.tools) if req.tools else None,
        }

        # Remove None fields
        data = {k: v for k, v in data.items() if v is not None}

        response = self._post_with_retry(api_url, json=data, headers=headers)

        try:
            response_data = response.json()
        except Exception as e:
            print(f"API Response: {response.text}")
            raise e
        content: List[LLMChatContentPartType] = []
        for part in response_data["candidates"][0]["content"]["parts"]:
            part: dict[str, dict | str]
            if "text" in part:
                content.append(LLMChatTextContent(text=part["text"]))
            elif "inlineData" in part:
                decoded_image_data = base64.b64decode(part["inlineData"]["data"])
                media = loop.run_until_complete(
                    self.media_manager.register_from_data(
                        data=decoded_image_data,
                        format=part["inlineData"]["mimeType"].removeprefix(
                            "image/"),
                        source="gemini response")
                )
                content.append(LLMChatImageContent(media_id=media))
            elif "functionCall" in part:
                # tool_call 部分 gemini 不返回call_id
                content.append(LLMToolCallContent(name=part["functionCall"]["name"], parameters=part["functionCall"].get("args", None)))
   
        return LLMChatResponse(
            model=req.model,
            usage=Usage(
                prompt_tokens=response_data["usageMetadata"].get(
                    "promptTokenCount"),
                cached_tokens=response_data["usageMetadata"].get(
                    "cachedContentTokenCount"),
                completion_tokens=sum([modality.get(
                    "tokenCount", 0) for modality in response_data.get("promptTokensDetails", [])]),
                total_tokens=response_data["usageMetadata"].get(
                    "totalTokenCount"),
            ),
            message=Message(
                content=content,
                role="assistant",
                finish_reason=response_data["candidates"][0].get("finishReason"),
                # content格式转好直接用就行
                tool_calls=resolve_function_call(content)
            ),
        )

    async def auto_detect_models(self) -> list[str]:
        api_url = f"{self.config.api_base}/models"
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(
                api_url, headers={"x-goog-api-key": self.config.api_key}
            ) as response:
                if response.status != 200:
                    self.logger.error(f"获取模型列表失败: {await response.text()}")
                    response.raise_for_status()
                response_data = await response.json()
                return [
                    model["name"].removeprefix("models/")
                    for model in response_data["models"]
                    if "generateContent" in model["supportedGenerationMethods"]
                ]

    def _post_with_retry(self, url: str, json: dict, headers: dict, retry_count: int = 3) -> requests.Response: # type: ignore
        for i in range(retry_count):
            try:
                response = requests.post(url, json=json, headers=headers)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if i == retry_count - 1:
                    print(
                        f"API Response: {response.text if 'response' in locals() else 'No response'}")
                    raise e
                else:
                    self.logger.warning(
                        f"Request failed, retrying {i+1}/{retry_count}: {e}")
