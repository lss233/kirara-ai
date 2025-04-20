from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from kirara_ai.im.message import IMMessage
from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.logger import get_logger
from kirara_ai.workflow.core.workflow import Workflow
from kirara_ai.workflow.core.workflow.registry import WorkflowRegistry

logger = get_logger("DispatchRule")

class SimpleDispatchRule(BaseModel):
    """简单规则，包含规则类型和配置"""
    type: str
    config: Dict[str, Any]

class RuleGroup(BaseModel):
    """规则组，包含多个简单规则和组合操作符"""
    operator: Literal["and", "or"] = "or"
    rules: List[SimpleDispatchRule]

class CombinedDispatchRule(BaseModel):
    """组合调度规则，支持复杂的规则组合"""
    rule_id: str
    name: str
    description: str = ""
    workflow_id: str
    priority: int = 5
    enabled: bool = True
    rule_groups: List[RuleGroup]  # 规则组之间是 AND 关系
    metadata: Dict[str, Any] = {}

    def match(self, message: IMMessage, workflow_registry: WorkflowRegistry, container: DependencyContainer) -> bool:
        """
        判断消息是否匹配该规则。
        规则组之间是 AND 关系，规则组内部根据 operator 决定是 AND 还是 OR 关系。
        """
        # 如果规则被禁用，直接返回 False
        if not self.enabled:
            return False

        # 所有规则组都必须匹配（AND 关系）
        for group in self.rule_groups:
            
            # 如果组内没有规则，视为匹配
            if len(group.rules) == 0:
                return True

            # 获取组内所有规则的匹配结果
            rule_results = []
            for rule in group.rules:
                try:
                    from ..rules.base import DispatchRule

                    # 创建具体的规则实例
                    rule_class = DispatchRule.get_rule_type(rule.type)
                    rule_instance = rule_class.from_config(
                        rule_class.config_class(**rule.config),
                        workflow_registry,
                        self.workflow_id,
                    )
                    rule_results.append(rule_instance.match(message, container))
                except Exception as e:
                    # 如果规则创建或匹配过程出错，视为不匹配
                    logger.error(f"Rule {rule.type} from config {rule.config} creation or matching failed: {e}")
                    continue

            # 根据操作符确定组的匹配结果
            if not rule_results:  # 如果组内没有有效规则，视为不匹配
                return False

            if group.operator == "and":
                if not all(rule_results):  # AND 关系：所有规则都必须匹配
                    return False
            else:  # operator == "or"
                if not any(rule_results):  # OR 关系：至少一个规则匹配
                    return False

        # 所有规则组都匹配成功
        return True

    def get_workflow(self, container: DependencyContainer) -> Optional[Workflow]:
        """获取该规则对应的工作流实例。"""
        workflow = container.resolve(WorkflowRegistry).get_workflow(self.workflow_id, container)
        return workflow
