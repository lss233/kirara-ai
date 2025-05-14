import asyncio
import json
from typing import Any, Dict, List, cast, Literal, TypedDict

import aiohttp
import requests
from pydantic import BaseModel, ConfigDict

import kirara_ai.llm.format.tool as tools
from kirara_ai.config.global_config import ModelConfig
from kirara_ai.llm.adapter import AutoDetectModelsProtocol, LLMBackendAdapter, LLMChatProtocol, LLMEmbeddingProtocol
from kirara_ai.llm.format.message import (LLMChatContentPartType, LLMChatImageContent, LLMChatMessage,
                                          LLMChatTextContent, LLMToolCallContent, LLMToolResultContent)
from kirara_ai.llm.format.request import LLMChatRequest, Tool
from kirara_ai.llm.format.response import LLMChatResponse, Message, Usage
from kirara_ai.llm.format.embedding import LLMEmbeddingRequest, LLMEmbeddingResponse
from kirara_ai.logger import get_logger
from kirara_ai.media import MediaManager
from kirara_ai.tracing import trace_llm_chat

from .utils import guess_openai_model, pick_tool_calls

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
                output = f"Error: {element.name}\n{output}"
            outputs.append({
                "role": "tool",
                "tool_call_id": element.id,
                "content": output,
            })
        return outputs
    else:
        parts: list[dict[str, Any]] = []
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

async def convert_llm_chat_message_to_openai_message(messages: list[LLMChatMessage], media_manager: MediaManager) -> list[dict]:
    # gather 必须先包一层异步函数，转化为协程对象， 否侧报错
    results = await asyncio.gather(*[convert_parts_factory(msg, media_manager) for msg in messages])
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


class OpenAIAdapterChatBase(LLMBackendAdapter, AutoDetectModelsProtocol, LLMChatProtocol):
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
        
        data = {
            "messages": convert_llm_chat_message_to_openai_message(req.messages, self.media_manager),
            "model": req.model,
            "frequency_penalty": req.frequency_penalty,
            "max_completion_tokens": req.max_tokens, # 最新的reference废除max_tokens，改为如上参数
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
    async def get_models(self) -> list[str]:
        api_url = f"{self.config.api_base}/models"
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(
                api_url, headers={"Authorization": f"Bearer {self.config.api_key}"}
            ) as response:
                response.raise_for_status()
                response_data = await response.json()
                return [model["id"] for model in response_data.get("data", [])]


    async def auto_detect_models(self) -> list[ModelConfig]:
        models = await self.get_models()
        all_models: list[ModelConfig] = []
        for model in models:
            guess_result = guess_openai_model(model)
            if guess_result is None:
                continue
            all_models.append(ModelConfig(id=model, type=guess_result[0].value, ability=guess_result[1]))
        return all_models
    
class EmbeddingData(TypedDict):
    object: Literal["embedding"]
    embedding: list[float]
    index: int

class EmbeddingResponse(TypedDict):
    # 用于描述类型定义
    object: Literal["list"]
    data: list[EmbeddingData]
    model: str
    usage: dict[Literal["prompt_tokens", "total_tokens"], int]

class OpenAIAdapter(OpenAIAdapterChatBase, LLMEmbeddingProtocol):
    def embed(self, req: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        """
        此为openai api嵌入式模型接口

        Tips: openai仅在 text-embedding-3 及以后模型中支持设定输出向量维度
        """
        
        api_url = f"{self.config.api_base}/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if len(req.inputs) > 2048:
            # text数组不能超过2048个元素，openai api限制
            raise ValueError("Text list has too many dimensions, max dimension is 2048")
        if any(isinstance(input, LLMChatImageContent) for input in req.inputs):
            # 未在api中发现多模态嵌入api, 等待后续更新
            raise ValueError("openai does not support multi-modal embedding")
        # mypy 类型检查修复，如果添加多模态请去除这个标注
        inputs = cast(list[LLMChatTextContent], req.inputs)
        data = {
            "text": [input.text for input in inputs],
            "model": req.model,
            "dimensions": req.dimension,
            "encoding_format": req.encoding_format
        }
        # 删除 None 字段
        data = {k: v for k, v in data.items() if v is not None}
        logger.debug(f"Request: {data}")
        response = requests.post(api_url, headers=headers, json=data)
        try:
            response.raise_for_status()
            response_data: EmbeddingResponse = response.json()
        except Exception as e:
            logger.error(f"Response: {response.text}")
            raise e
        logger.debug(f"Response: {response_data}")
        return LLMEmbeddingResponse(
            vectors=[data["embedding"] for data in response_data["data"]],
            usage=Usage(
                prompt_tokens=response_data["usage"].get("prompt_tokens", 0),
                total_tokens=response_data["usage"].get("total_tokens", 0)
            )
        )