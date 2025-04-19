from typing import Any, List, Optional

from pydantic import BaseModel

from kirara_ai.llm.format.message import LLMChatMessage
from kirara_ai.llm.format.tool import Tool


class ResponseFormat(BaseModel):
    type: Optional[str] = None

class LLMChatRequest(BaseModel):
    """
    Attributes:
        tool_choice (Union[dict, Literal["auto", "any", "none"]]): 
            "
            注意由于大模型对于这个接口实现不同，本次暂不实现tool_choice的功能。
            tool_choice这个参数告诉llmMessage应该如何选择调用的工具。
            "
    """
    
    messages: List[LLMChatMessage] = []
    model: Optional[str] = None
    frequency_penalty: Optional[int] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[int] = None
    response_format: Optional[ResponseFormat] = None
    stop: Optional[Any] = None
    stream: Optional[bool] = None
    stream_options: Optional[Any] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    # 规范tool传递
    tools: Optional[list[Tool]] = None 
    # tool_choice各家目前标准不尽相同，暂不向用户提供更改这个值的选项
    tool_choice: Optional[Any] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[Any] = None