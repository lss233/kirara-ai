from typing import List, Literal, Union

from pydantic import BaseModel


class ImageURL(BaseModel):
    url: str

class LLMChatTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class LLMChatImageContent(BaseModel):
    type: Literal["image"] = "image"
    media_id: str

class LLMChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: List[Union[LLMChatTextContent, LLMChatImageContent]]
