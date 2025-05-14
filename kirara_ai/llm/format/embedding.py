from typing import Literal, Optional
from pydantic import BaseModel

from .message import LLMChatTextContent, LLMChatImageContent
from .response import Usage

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
        input_type (Optional[InputType]): 输入类型, 归属于voyage_adapter的独有字段
        truncate (Optional[bool]): 是否自动截断超长文本, 以适应llm上下文长度上限。
        output_type (Optional[OutputType]): 向量内部应该使用哪种数据类型. 一般默认float
    """
    inputs: list[InputUnionType]
    model: str
    dimension: Optional[int] = None
    encoding_format: Optional[FormatType] = None
    input_type: Optional[InputType] = None
    truncate: Optional[bool] = None
    output_type: Optional[OutputType] = None

vector = list[float | int] # 后续可能需要使用numpy库进行精确的数据类型标注, 暂时未处理base64的返回模式
class LLMEmbeddingResponse(BaseModel):
    """
    向量维度请使用len(vector)自行计算。
    Attributes:
        vectors: list[vector]
        usage: Optional[Usage] = None
    """
    vectors: list[vector]
    usage: Optional[Usage] = None