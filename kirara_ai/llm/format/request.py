from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel

from kirara_ai.llm.format.message import LLMChatMessage


class ToolParameters(BaseModel):
    """
    规范化工具参数的格式

    Attributes:
        type (Literal["object"]): 参数的类型
        properties (dict): 工具属性，参考 openai api 的规范
        required (list[str]): 必填参数的名称列表
        additionalProperties (Optional[bool]): 是否允许额外的键值对
    """
    type: Literal["object"] = "object"
    properties: dict
    required: list[str]
    additionalProperties: Optional[bool] = False

class Tool(BaseModel):
    """
    这是传递给 llm 的工具信息

    Attributes:
        type (Optional[Literal["function"]]): 工具的类型
        name (str): 工具的名称
        description (str): 工具的描述
        parameters (ToolParameters): 工具的参数
        strict (Optional[bool]): 是否严格调用, openai api专属
    """
    type: Optional[Literal["function"]] = "function"
    name: str
    description: str
    parameters: Union[ToolParameters, dict]
    strict: Optional[bool] = False

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