import asyncio

import aiohttp
import requests
from pydantic import BaseModel, ConfigDict

from kirara_ai.llm.adapter import AutoDetectModelsProtocol, LLMBackendAdapter
from kirara_ai.llm.format.message import LLMChatImageContent, LLMChatMessage, LLMChatTextContent
from kirara_ai.llm.format.request import LLMChatRequest
from kirara_ai.llm.format.response import LLMChatResponse
from kirara_ai.logger import get_logger
from kirara_ai.media import MediaManager

SAFETY_SETTINGS = [{
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
},{
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
},{
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
},{
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
},{
    "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
    "threshold": "BLOCK_NONE"
}]

# POST 模式支持最大 20 MB 的 inline data
INLINE_LIMIT_SIZE = 1024 * 1024 * 20

class GeminiConfig(BaseModel):
    api_key: str
    api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    model_config = ConfigDict(frozen=True)


async def convert_llm_chat_message_to_gemini_message(msg: LLMChatMessage, media_manager: MediaManager) -> dict:
    parts = []
    for element in msg.content:
        if isinstance(element, LLMChatTextContent):
            parts.append({"text": element.text})
        elif isinstance(element, LLMChatImageContent):
            media = media_manager.get_media(element.media_id)
            parts.append({
                "inline_data": {
                    "mime_type": media.mime_type,
                    "data": await media.get_base64()
                }
            })

    return {
        "role": "model" if msg.role == "assistant" else "user",
        "parts": parts
    }


class GeminiAdapter(LLMBackendAdapter, AutoDetectModelsProtocol):
    
    media_manager: MediaManager
    
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.logger = get_logger("GeminiAdapter")

    def chat(self, req: LLMChatRequest) -> LLMChatResponse:
        api_url = f"{self.config.api_base}/models/{req.model}:generateContent"
        headers = {
            "x-goog-api-key": self.config.api_key,
            "Content-Type": "application/json",
        }
        
        # create a new asyncio loop to run the convert_llm_chat_message_to_gemini_message function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # use asyncio gather to run the convert_llm_chat_message_to_gemini_message function
        contents = loop.run_until_complete(
            asyncio.gather(
                *[convert_llm_chat_message_to_gemini_message(msg, self.media_manager) for msg in req.messages]
            )
        )

        data = {
            "contents": contents ,
            "generationConfig": {
                "temperature": req.temperature,
                "topP": req.top_p,
                "topK": 40,
                "maxOutputTokens": req.max_tokens,
                "stopSequences": req.stop,
            },
            "safetySettings": SAFETY_SETTINGS,
        }

        self.logger.debug(f"Contents: {data['contents']}")
        # Remove None fields
        data = {k: v for k, v in data.items() if v is not None}
        response = requests.post(api_url, json=data, headers=headers)
        try:
            response.raise_for_status()
            response_data = response.json()
        except Exception as e:
            print(f"API Response: {response.text}")
            raise e
        print(response_data)

        # Transform Gemini response format to match expected LLMChatResponse format
        transformed_response = {
            "id": response_data.get("promptFeedback", {}).get("blockReason", ""),
            "object": "chat.completion",
            "created": 0,  # Gemini doesn't provide creation timestamp
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_data["candidates"][0]["content"]["parts"][
                            0
                        ]["text"],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # Gemini doesn't provide token counts
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        return LLMChatResponse(**transformed_response)

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
