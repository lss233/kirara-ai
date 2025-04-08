import json
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, field_validator, model_validator
from typing_extensions import Self

from mcp.types import EmbeddedResource, ImageContent, TextContent

RoleType = Literal["system", "user", "assistant"]
ResultTypes = Union[TextContent, ImageContent, EmbeddedResource]

class LLMChatTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class LLMChatImageContent(BaseModel):
    type: Literal["image"] = "image"
    media_id: str

class LLMToolCallContent(BaseModel):
    """
    这是模型请求工具的消息内容,
    模型强相关内容，如果你 message 或者 memory 内包含了这个内容，请保证调用同一个 model
    此部分 role 应该归属于"assistant"
    """
    type: Literal["tool_call"] = "tool_call"
    id: Optional[str] = None
    name: str
    parameters: Optional[dict] = None

    @classmethod
    @field_validator("parameters", mode="before")
    def convert_parameters_to_dict(cls, v: Optional[Union[str, dict]]) -> Optional[dict]:
        if isinstance(v, str):
            return json.loads(v)
        return v

class LLMToolResultContent(BaseModel):
    """
    这是工具回应的消息内容,
    模型强相关内容，如果你 message 或者 memory 内包含了这个内容，请保证调用同一个 model
    此部分 role 应该对应 "tool"
    """
    type: Literal["tool_result"] = "tool_result"
    id: Optional[str] = None
    name: str
    # 统一传递mcp.types.CallToolResult交由adapter自行处理, str是仅发生错误时才允许的类型
    content: Union[list[ResultTypes],str]
    isError: bool = False

    @model_validator(mode="after")
    def check_content_type(self) -> Self:
        if self.isError and not isinstance(self.content, str):
            raise ValueError("content must be a str, when isError is True")
        return self

LLMChatContentPartType = Union[LLMChatTextContent, LLMChatImageContent, LLMToolCallContent, LLMToolResultContent]
RoleTypes = Literal["user", "assistant", "system", "tool"]

class LLMChatMessage(BaseModel):
    """
    当 role 为 "tool" 时，content 内部只能为 list[LLMToolResultContent]
    """
    content: list[LLMChatContentPartType]
    role: RoleTypes

    @model_validator(mode="after")
    def check_content_type(self) -> Self:
        # 此装饰器将在 model 实例化后执行，`mode = "after"`
        # 用于检查 content 字段的类型是否符合 role 要求
        match self.role:
            case "user" | "assistant" | "system":
                if not all(any(isinstance(element, content_type) for content_type in [LLMChatTextContent, LLMChatImageContent, LLMToolCallContent]) for element in self.content):
                    raise ValueError(f"content must be a list of LLMChatContentPartType, when role is {self.role}")
            case "tool":
                if not all(isinstance(element, LLMToolResultContent) for element in self.content):
                    raise ValueError("content must be a list of LLMToolResultContent, when role is 'tool'")
        return self