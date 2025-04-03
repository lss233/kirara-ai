import asyncio
from typing import List

import aiohttp
import requests
import json
from pydantic import  ConfigDict
from typing import cast, Union, Optional

from kirara_ai.config.global_config import LLMBackendConfig
from kirara_ai.llm.adapter import AutoDetectModelsProtocol, LLMBackendAdapter
from kirara_ai.llm.format.message import LLMChatImageContent, LLMChatMessage, LLMChatTextContent, LLMChatContentPartType, LLMToolCallContent, LLMToolResultContent
from kirara_ai.llm.format.request import LLMChatRequest
from kirara_ai.llm.format.response import LLMChatResponse, Message, ToolCall, Usage, Function
from kirara_ai.logger import get_logger
from kirara_ai.media import MediaManager
from kirara_ai.tracing import trace_llm_chat

logger = get_logger("OpenAIAdapter")

async def convert_parts_factory(messages: LLMChatMessage, media_manager: MediaManager) -> list[dict]:
    if messages.role == "tool":
        # typing.cast 指定类型，避免mypy报错
        results = cast(list[LLMToolResultContent], messages.content)
        # 保证 content 为一个字符串
        return [{"role": "tool", "tool_call_id": result.id, "content": str(result.content)} for result in results]
    else:
        parts = []
        elements = cast(list[LLMChatContentPartType], messages.content)
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
                # 忽略tool_call_content，openai api不需要。
                # 保留这个判断分支，防止openai api接口出现变动。
                continue
        return [{"role": messages.role, "content": parts}]

def convert_llm_chat_message_to_openai_message(messages: list[LLMChatMessage], media_manager: MediaManager, loop: asyncio.AbstractEventLoop) -> list[dict]:
    results = loop.run_until_complete(
        asyncio.gather(*[convert_parts_factory(msg, media_manager) for msg in messages])
    )
    # 扁平化结果, 展开所有列表
    return [item for sublist in results for item in sublist]

def resolve_tool_calls_from_response(tool_calls: Optional[list[dict[str, Union[str, dict[str, str]]]]]):
    if tool_calls is None:
        return None
    else:
        return [ToolCall(
            id=call["id"],
            type=call["type"],
            function=Function(
                name=call["function"]["name"],
                # openai api 的 arguments 值是一个长得像dict的字符串
                arguments= json.loads(call["function"]["arguments"]) if call["function"].get("arguments", None) else None
            )
        ) for call in tool_calls]

class OpenAIConfig(LLMBackendConfig):
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
            "tools": [tool.model_dump() for tool in req.tools] if req.tools else None,
            "tool_choice": "auto",
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

        choices = response_data.get("choices", [{}])
        first_choice = choices[0] if choices else {}
        message: dict = first_choice.get("message", {})
        
        # 检测tool_calls字段是否存在和是否不为None. tool_call时content字段无有效信息，暂不记录
        if tool_calls := message.get("tool_calls", None):
            content = [LLMToolCallContent(
                id=call["id"],
                name=call["function"]["name"],
                parameters=call["function"].get("parameters", None)
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
                # tool_calls=[
                #     ToolCall(
                #         model = "openai",
                #         id=tool_call["id"], 
                #         type=tool_call["type"],
                #         function=Function(name = tool_call["function"]["name"], arguments=tool_call["function"].get("arguments", None))
                #     ) for tool_call in message.get("tool_calls")
                # ] if message.get("tool_calls", None) else None,
                tool_calls = resolve_tool_calls_from_response(response_data.get("tool_calls", None)),
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