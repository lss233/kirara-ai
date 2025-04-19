from typing import Any, List, Optional, Literal

from pydantic import BaseModel

from kirara_ai.llm.format.message import LLMChatMessage, LLMChatTextContent, LLMChatImageContent


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
    parameters: ToolParameters
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

FormatType = Literal["base64"]
OutputType = Literal["float", "int8", "uint8", "binary", "ubinary"]
InputType = Literal["string", "query", "document"]
InputUnionType = LLMChatTextContent | LLMChatImageContent

class LLMEmbeddingRequest(BaseModel):
    """
    此模型用于规范embedding请求的格式
    Tips: 各大模型向量维度以及向量转化函数不同，因此当你用于向量数据库时，请确保存储和检索使用同一个模型，并确保模型向量一致（部分模型支持同一模型设置向量维度）
    Note: 注意一下字段为混合字段, 部分字段在部分模型中不起作用, 请参照对应ap文档传递参数。

    Attributes:
        text (list[str | Image]): 待转化为向量的文本或图片列表
        model (str): 使用的embedding模型名
        dimensions (Optional[int]): embedding向量的维度
        encoding_format (Optional[FormatType]): embedding的编码格式。推荐不设置该字段, 方便直接输入数据库
        user (Optional[str]): 用户名, openai可选字段目前不知道有什么用
        input_type (Optional[InputType]): 输入类型, 归属于voyage_adapter的独有字段
        truncate (Optional[bool]): 是否自动截断超长文本, 以适应llm上下文长度上限。
        output_type (Optional[OutputType]): 向量内部应该使用哪种数据类型. 一般默认float
    """
    inputs: list[InputUnionType]
    model: str
    dimension: Optional[int] = None
    encoding_format: Optional[FormatType] = None
    user: Optional[str] = None
    input_type: Optional[InputType] = None
    truncate: Optional[bool] = None
    output_type: Optional[OutputType] = None