import json
from typing import Any, Callable, Coroutine, Generic, List, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

ModelTypes = Literal["openai", "gemini", "claude", "ollama"]

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class MediaContent(BaseModel):
    type: Literal["media"] = "media"
    media_id: str
    mime_type: str
    data: bytes

ToolResponseTypes = List[Union[TextContent, MediaContent]]

class LLMToolResultContent(BaseModel):
    """
    这是工具回应的消息内容,
    模型强相关内容，如果你 message 或者 memory 内包含了这个内容，请保证调用同一个 model
    此部分 role 应该对应 "tool"
    """
    type: Literal["tool_result"] = "tool_result"
    # call id，对应 LLMToolCallContent 的 id
    id: str
    name: str
    # 各家工具要求返回的content格式不同. 等待后续规范化。
    content: ToolResponseTypes
    isError: bool = False

class Function(BaseModel):
    # 工具名称
    name: str
    # 这个字段类似于 python 的关键子参数，你可以直接使用`**arguments`
    arguments: Optional[dict] = None

    @classmethod
    @field_validator("arguments", mode="before")
    def convert_arguments(cls, v: Optional[Union[str, dict]]) -> Optional[dict]:
        return json.loads(v) if isinstance(v, str) else v

class ToolCall(BaseModel):
    # call id，对应 LLMToolCallContent 的 id
    id: str
    # type这个字段目前不知道有什么用
    type: Optional[str] = None
    function: Function
    
T = TypeVar('T', bound=Callable)

ToolInvokeFunc = Callable[[ToolCall], Coroutine[Any, Any, "LLMToolResultContent"]]

class CallableWrapper(Generic[T]):
    """包装可调用对象的类，在深拷贝时返回None"""
    def __init__(self, func: T):
        self.func = func
    
    def __call__(self, *args, **kwargs) -> Coroutine[Any, Any, "LLMToolResultContent"]:
        return self.func(*args, **kwargs)
    
    def __deepcopy__(self, memo):
        # 深拷贝时保持原始引用而不是尝试复制函数
        return self


class ToolInputSchema(BaseModel):
    """
    工具输入参数的格式，遵循 JSON Schema 的规范

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
    传递给 LLM 的工具信息

    Attributes:
        type (Optional[Literal["function"]]): 工具的类型
        name (str): 工具的名称
        description (str): 工具的描述
        parameters (ToolInputSchema): 工具的参数格式
        strict (Optional[bool]): 是否严格调用, openai api专属
        invokeFunc (Optional[Callable]): 工具对应的执行函数，仅在调用时使用，不参与序列化
    """
    type: Optional[Literal["function"]] = "function"
    name: str
    description: str
    parameters: Union[ToolInputSchema, dict]
    strict: Optional[bool] = False
    
    invokeFunc: CallableWrapper[ToolInvokeFunc] = Field(exclude=True)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_serializer("invokeFunc")
    def serialize_invoke_func(self, invoke_func: CallableWrapper[ToolInvokeFunc]) -> str:
        return "..."
