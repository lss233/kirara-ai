from typing import List, Optional, Literal

from pydantic import BaseModel

from kirara_ai.llm.format.message import LLMChatMessage


ModelTypes = Literal["openai", "gemini", "claude", "ollama"]

class Function(BaseModel):
    name: Optional[str] = None
    # 这个字段类似于 python 的关键子参数，你可以直接使用`**arguments`
    arguments: Optional[dict] = None


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
