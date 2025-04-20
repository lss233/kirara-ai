import random

from pydantic import Field

from kirara_ai.im.adapter import IMAdapter
from kirara_ai.im.manager import IMManager
from kirara_ai.im.message import IMMessage
from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.workflow.core.workflow.registry import WorkflowRegistry

from .base import DispatchRule, RuleConfig


class RandomChanceRuleConfig(RuleConfig):
    """随机概率规则配置"""
    chance: int = Field(
        default=50, ge=0, le=100, title="随机概率", description="随机概率，范围为0-100"
    )

class RandomChanceMatchRule(DispatchRule):
    """根据随机概率匹配的规则"""
    config_class = RandomChanceRuleConfig
    type_name = "random"

    def __init__(self, chance: int, workflow_registry: WorkflowRegistry, workflow_id: str):
        super().__init__(workflow_registry, workflow_id)
        self.chance = chance

    def match(self, message: IMMessage) -> bool:
        print(f"Random chance: {self.chance}")
        print(f"Random number: {random.random()}")
        return random.random() * 100 < self.chance

    def get_config(self) -> RandomChanceRuleConfig:
        return RandomChanceRuleConfig(chance=self.chance)

    @classmethod
    def from_config(
        cls, config: RandomChanceRuleConfig, workflow_registry: WorkflowRegistry, workflow_id: str
    ) -> "RandomChanceMatchRule":
        return cls(config.chance, workflow_registry, workflow_id)

class IMInstanceMatchRuleConfig(RuleConfig):
    """IM实例匹配规则配置"""
    im_instance: str = Field(title="IM实例名称", description="配置后，只有当消息来自指定的IM实例时，才会触发工作流")

class IMInstanceMatchRule(DispatchRule):
    """根据IM实例匹配的规则"""
    config_class = IMInstanceMatchRuleConfig
    type_name = "im_instance"

    def __init__(self, im_instance: str, workflow_registry: WorkflowRegistry, workflow_id: str):
        super().__init__(workflow_registry, workflow_id)
        self.im_instance = im_instance

    def match(self, message: IMMessage, container: DependencyContainer) -> bool:
        adapter = container.resolve(IMAdapter)
        im_manager = container.resolve(IMManager)
        return im_manager.get_adapter(self.im_instance) == adapter

    def get_config(self) -> IMInstanceMatchRuleConfig:
        return IMInstanceMatchRuleConfig(im_instance=self.im_instance)

    @classmethod
    def from_config(
        cls, config: IMInstanceMatchRuleConfig, workflow_registry: WorkflowRegistry, workflow_id: str
    ) -> "IMInstanceMatchRule":
        return cls(config.im_instance, workflow_registry, workflow_id)


class FallbackMatchRule(DispatchRule):
    """默认的兜底规则，总是匹配"""
    config_class = RuleConfig
    type_name = "fallback"

    def __init__(self, workflow_registry: WorkflowRegistry, workflow_id: str):
        super().__init__(workflow_registry, workflow_id)
        self.priority = 0  # 兜底规则优先级最低

    def match(self, message: IMMessage, container: DependencyContainer) -> bool:
        return True

    def get_config(self) -> RuleConfig:
        return RuleConfig()

    @classmethod
    def from_config(
        cls, config: RuleConfig, workflow_registry: WorkflowRegistry, workflow_id: str
    ) -> "FallbackMatchRule":
        return cls(workflow_registry, workflow_id) 