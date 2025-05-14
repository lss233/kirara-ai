from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional, Union
from typing_extensions import Self

class ImageUrl(BaseModel):
    url: str
    detail: Optional[str] = "auto"

class InputAudio(BaseModel):
    data: str # base64 format
    format: Literal["wav", "mp3"]

class File(BaseModel):
    file_data: Optional[str] = Field(description="The base64 encoded file data, used when passing the file to the model as a string.")
    file_id: Optional[str] = Field(description="The ID of an uploaded file to use as input.")
    filename: Optional[str] = Field(description="The name of the file, used when passing the file to the model as a string.")

class TextContent(BaseModel):
    text: str
    type: Literal["text"]

class ImageContent(BaseModel):
    image_url: ImageUrl
    type: Literal["image_url"]

class AudioContent(BaseModel):
    input_audio: InputAudio
    type: Literal["input_audio"]

class FileContent(BaseModel):
    file: File
    type: Literal["file"]

class RefusalContent(BaseModel):
    refusal: str
    type: Literal["refusal"]

UserUnionContent = Union[TextContent, ImageContent, AudioContent, FileContent]
AssistantUnionContent = Union[TextContent, RefusalContent]

class Function(BaseModel):
    arguments: str
    name: str

class ToolCall(BaseModel):
    function: Function
    id: str
    type: Literal["function"]

class DeveloperMessage(BaseModel):
    role: Literal["developer"]
    name: Optional[str] = None
    content: list[TextContent] | str

class SystemMessage(BaseModel):
    role: Literal["system"]
    name: Optional[str] = None
    content: list[TextContent] | str

class UserMessage(BaseModel):
    role: Literal["user"]
    name: Optional[str] = None
    content: list[UserUnionContent] | str

class AssistantMessage(BaseModel):
    role: Literal["assistant"]
    audio: Optional[dict[Literal["id"], str]] = None
    content: list[AssistantUnionContent] | str
    name: Optional[str]
    refusal: Optional[str] = None
    tool_calls: Optional[list[ToolCall]]

class ToolMessage(BaseModel):
    role: Literal["tool"]
    content: list[TextContent] | str
    tool_call_id: str

UnionMessage = Union[DeveloperMessage, SystemMessage, UserMessage, AssistantMessage, ToolMessage]

class TopAudio(BaseModel):
    """api_reference中最顶层的audio类型, 定义llm的音频输出"""
    format: Literal["wav", "mp3", "flac", "opus", "pcm16"]
    voice: Literal["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]

class StaticContent(BaseModel):
    type: Literal["content"]
    content: list[TextContent] | str

class ChatRequest(BaseModel):
    model: Literal["mock_chat"]
    messages: list[UnionMessage]
    audio: Optional[TopAudio] = None
    frequency_penalty: float = Field(default=0, ge=-2.0, le=2.0)
    logit_bias: Optional[dict] = None
    logprobs: Optional[bool] = False
    max_completion_tokens: Optional[int] = None
    metadata: Optional[dict] = None
    modalities: Optional[list] = None
    n: Optional[int] = 1
    parallel_tool_calls: Optional[bool] = True
    prediction: Optional[StaticContent] = None
    presence_penalty: Optional[float] = Field(default=0, ge=-2.0, le=2.0)
    reasoning_effort: Optional[str] = "medium"
    response_format: Optional[dict] = None
    seed: Optional[int] = None
    service_tier: Optional[str] = "auto"
    stop: Optional[str|list] = None
    store: Optional[bool] = False
    stream: Optional[bool] = False
    stream_options: Optional[dict] = None
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    tool_choice: Optional[str] = None
    tools: Optional[list] = None
    top_logprobs: Optional[int] = Field(ge=0, le=20)
    top_p: Optional[float] = 1
    user: Optional[str] = None
    web_search_options: Optional[dict] = None

    @model_validator(mode="after")
    def validate_top_logprobs(self) -> Self:
        # api_reference要求
        if self.top_logprobs and not self.logprobs:
            raise ValueError("top_logprobs can only be used with logprobs=True")
        return self


class EmbeddingRequest(BaseModel):
    text: list[str] | str
    model: Literal["mock_embedding", "text-embedding-ada-002"]
    dimensions: Optional[int] = None
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None

    @model_validator(mode="after")
    def custom_validate(self) -> Self:
        if self.dimensions and self.model in ["text-embedding-ada-002"]:
            raise ValueError("The number of dimensions the resulting output embeddings should have. Only supported in text-embedding-3 and later models.")
        return self
        