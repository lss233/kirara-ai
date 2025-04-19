from typing import List, Optional, Literal, Union
import json
from pydantic import BaseModel, field_validator

from kirara_ai.llm.format.message import LLMChatMessage


ModelTypes = Literal["openai", "gemini", "claude", "ollama"]

class Function(BaseModel):
    name: Optional[str] = None
    # 这个字段类似于 python 的关键子参数，你可以直接使用`**arguments`
    arguments: Optional[dict] = None

    @classmethod
    @field_validator("arguments", mode="before")
    def convert_arguments(cls, v: Optional[Union[str, dict]]) -> Optional[dict]:
        if isinstance(v, str):
            return json.loads(v)
        else:
            return v

class ToolCall(BaseModel):
    id: Optional[str] = None
    # type这个字段目前不知道有什么用
    type: Optional[str] = None
    # 此参数用于向后端传递响应的模型类型，方便后端tool_result返回类型正确的content字段
    model: Optional[ModelTypes] = "openai"
    function: Optional[Function] = None

class Message(LLMChatMessage):
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None

class LLMChatResponse(BaseModel):
    model: Optional[str] = None
    usage: Optional[Usage] = None
    message: Message

vector = list[float | int] # 后续可能需要使用numpy库进行精确的数据类型标注
class LLMEmbeddingResponse(BaseModel):
    """
    向量维度请使用len(vector)自行计算。
    Attributes:
        vectors: list[vector]
        usage: Optional[Usage] = None
    """
    vectors: list[vector]
    usage: Optional[Usage] = None