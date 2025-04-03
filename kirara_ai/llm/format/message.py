from typing import Literal, List, Optional, Union, Any, Self

from pydantic import BaseModel, model_validator

RoleType = Literal["system", "user", "assistant"]

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
    # 有些model不会返回 call_id ，点名 gemini
    id: Optional[str] = None
    name: str
    # tool可能没有参数。
    parameters: Optional[dict | str] = None

class LLMToolResultContent(BaseModel):
    """
    这是工具回应的消息内容,
    模型强相关内容，如果你 message 或者 memory 内包含了这个内容，请保证调用同一个 model
    此部分 role 应该对应 "tool"
    """
    type: Literal["tool_result"] = "tool_result"
    # 为与 gemini 兼容，此处 id 改为 Optional. 因为 gemini 回应中没有 call_id.
    id: Optional[str] = None
    name: str
    # 各家工具要求返回的content格式不同. 等待后续规范化。
    content: Any

LLMChatContentPartType = Union[LLMChatTextContent, LLMChatImageContent, LLMToolCallContent]
RoleTypes = Literal["user", "assistant", "system", "tool"]

class LLMChatMessage(BaseModel):
    """
    当 role 为 "tool" 时，content 内部只能为 list[LLMToolResultContent]
    """
    content: Union[List[LLMChatContentPartType], List[LLMToolResultContent]]
    role: RoleTypes

    @model_validator(mode="after")
    def check_content_type(self) -> Self:
        # 此装饰器将在 model 实例化后执行，`mode = "after"`
        # 用于检查 content 字段的类型是否符合 role 要求
        match self.role:
            case "user" | "assistant" | "system":
                if not all(isinstance(element, LLMChatContentPartType) for element in self.content):
                    raise ValueError(f"content must be a list of LLMChatContentPartType, when role is {self.role}")
            case "tool":
                if not all(isinstance(element, LLMToolResultContent) for element in self.content):
                    raise ValueError("content must be a list of LLMToolResultContent, when role is 'tool'")
        return self