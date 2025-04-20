from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Type

from pydantic import BaseModel

from kirara_ai.im.message import IMMessage
from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.workflow.core.workflow import Workflow
from kirara_ai.workflow.core.workflow.registry import WorkflowRegistry


class RuleConfig(BaseModel):
    """规则配置的基类"""

class DispatchRule(ABC):
    """
    工作流调度规则的抽象基类。
    用于定义如何根据消息内容选择合适的工作流进行处理。
    """
    # 类变量，用于规则类型注册
    rule_types: ClassVar[Dict[str, Type["DispatchRule"]]] = {}
    config_class: ClassVar[Type[RuleConfig]]
    type_name: ClassVar[str]

    def __init__(self, workflow_registry: WorkflowRegistry, workflow_id: str):
        """初始化调度规则。"""
        self.workflow_registry = workflow_registry
        self.rule_id: str = ""
        self.name: str = ""
        self.description: str = ""
        self.priority: int = 5  # 默认优先级为5
        self.enabled: bool = True  # 是否启用
        self.metadata: Dict[str, Any] = {}  # 元数据
        self.workflow_id: str = workflow_id

    @abstractmethod
    def match(self, message: IMMessage, container: DependencyContainer) -> bool:
        """判断消息是否匹配该规则。"""

    def get_workflow(self, container: DependencyContainer) -> Workflow:
        """获取该规则对应的工作流实例。"""
        workflow = self.workflow_registry.get(self.workflow_id, container)
        assert isinstance(workflow, Workflow)
        return workflow

    @classmethod
    def register_rule_type(cls, rule_class: Type["DispatchRule"]):
        """注册规则类型"""
        cls.rule_types[rule_class.type_name] = rule_class

    @classmethod
    def get_rule_type(cls, type_name: str) -> Type["DispatchRule"]:
        """获取规则类型"""
        if type_name not in cls.rule_types:
            raise ValueError(f"Unknown rule type: {type_name}")
        return cls.rule_types[type_name]

    @abstractmethod
    def get_config(self) -> RuleConfig:
        """获取规则配置"""

    @classmethod
    @abstractmethod
    def from_config(
        cls, config: RuleConfig, workflow_registry: WorkflowRegistry, workflow_id: str
    ) -> "DispatchRule":
        """从配置创建规则实例"""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id='{self.rule_id}', priority={self.priority}, enabled={self.enabled})" 