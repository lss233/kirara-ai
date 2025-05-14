from typing import Optional
from typing_extensions import Self
from pydantic import BaseModel, model_validator

from .response import Usage

class LLMReRankRequest(BaseModel):
    """
    ReRanker: 重排器是一个重要的处理方案, 通常见于 EsSearch 的优化方案中。
    本接口是适用于 LLM 的重排器的请求模型一般与嵌入式式模型组合使用提高向量搜索准确率。

    传入一系列原始文档和一个查询语句，返回器相似度数值。
    Attributes:
        query: 原始查询语句
        documents: 文档列表, 包含文档的文本内容。每个文档转化为一个 string 类型传递。
        model: 重排模型的名称。为保证准确性，本实现将禁止自动选择模型。
        top_k: 返回最相似的 {top_k} 个文档。如果没有指定，将返回所有文档的重排序结果。
            Tips: 如果你决定不返回原始文档，那么不要设置这个选项。会丢失文本与相似度的关联。
        return_documents: 是否返回原始文档内容。
        truncation: 文档和查询语句是否允许被截断以适应模型最大上下文。
        sort: 是否按照结果的相似度得分进行排序？ 默认不进行
            Tips: 当return_documents为False时，若sort为True，则抛出异常。
    """
    query: str
    documents: list[str]
    model: str
    top_k: Optional[int] = None
    return_documents: Optional[bool] = None
    truncation: Optional[bool] = None
    sort: Optional[bool] = False

    @model_validator(mode="after") # mode 为 before 时其用法与after完全不同，注意看官网文档
    # 这里不用after是为了等pydantic赋值默认值后检查
    def check(self) -> Self:
        if self.sort and not self.return_documents:
            raise ValueError("Cannot sort server responses when return_documents is False.")
        return self
    
class ReRankerContent(BaseModel):
    """
    ReRanker 的内容模型。

    Attributes:
        document: 原始文档内容。
        score: 文档的相似度分数。
    """

    document: Optional[str] = None
    score: float

class LLMReRankResponse(BaseModel):
    """
    ReRanker 的返回模型。

    Attributes:
        contents (list[ReRankerContent]): 返回的排序信息, 如果启用排序，默认降序排列。 Note: 当且仅当return_documents为True时才允许启用排序。
        usage (Usage): token 使用情况, 一个pydantic.BaseModel的子类。
        sort (bool): 是否按照结果的相似度排序？将其设置为字段方便后续接口检查是否经过排序(方便debug)。其应该由request的sort字段赋值。
    """

    contents: list[ReRankerContent]
    usage: Usage
    sort: bool

    @model_validator(mode="after") # 当mode为after时，其发生在class实例化完成后，所以其为实例方法
    def sort_content(self) -> Self:
        if self.sort:
            self.contents = sorted(self.contents, key= lambda x: x.score, reverse=True)
        return self