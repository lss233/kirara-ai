import os
from typing import Any, Dict, List, Optional

from ruamel.yaml import YAML

from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.logger import get_logger
from kirara_ai.workflow.core.workflow.registry import WorkflowRegistry

from .models.dispatch_rules import CombinedDispatchRule, RuleGroup, SimpleDispatchRule
from .rules.base import DispatchRule
from .rules.message_rules import BotMentionMatchRule, KeywordMatchRule, PrefixMatchRule, RegexMatchRule
from .rules.sender_rules import ChatSenderMatchRule, ChatSenderMismatchRule, ChatTypeMatchRule
from .rules.system_rules import FallbackMatchRule, IMInstanceMatchRule, RandomChanceMatchRule


class DispatchRuleRegistry:
    """调度规则注册表，管理调度规则的加载和注册"""

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.workflow_registry = container.resolve(WorkflowRegistry)
        self.rules: Dict[str, CombinedDispatchRule] = {}
        self.logger = get_logger("DispatchRuleRegistry")
        self.rules_dir = "data/dispatch_rules"

    def register(self, rule: CombinedDispatchRule):
        """注册一个调度规则"""
        if not rule.rule_id:
            raise ValueError("Rule must have an ID")
        self.rules[rule.rule_id] = rule
        self.logger.info(f"Registered dispatch rule: {rule}")

    def get_rule(self, rule_id: str) -> Optional[CombinedDispatchRule]:
        """获取指定ID的规则"""
        return self.rules.get(rule_id)

    def get_all_rules(self) -> List[CombinedDispatchRule]:
        """获取所有已注册的规则"""
        return list(self.rules.values())

    def get_active_rules(self) -> List[CombinedDispatchRule]:
        """获取所有已启用的规则，按优先级降序排序"""
        active_rules = [rule for rule in self.rules.values() if rule.enabled]
        return sorted(active_rules, key=lambda x: x.priority, reverse=True)

    def create_rule(self, rule: CombinedDispatchRule) -> CombinedDispatchRule:
        """创建并注册一个新规则"""
        # 获取工作流构建器
        workflow_builder = self.workflow_registry.get(rule.workflow_id)
        if not workflow_builder:
            raise ValueError(f"Workflow {rule.workflow_id} not found")

        # 注册规则
        self.register(rule)
        return rule

    def update_rule(
        self, rule_id: str, rule: CombinedDispatchRule
    ) -> CombinedDispatchRule:
        """更新现有规则"""
        if rule_id not in self.rules:
            raise ValueError(f"Rule {rule_id} not found")

        # 更新规则
        self.register(rule)
        return rule

    def delete_rule(self, rule_id: str):
        """删除规则"""
        if rule_id not in self.rules:
            raise ValueError(f"Rule {rule_id} not found")
        del self.rules[rule_id]

    def enable_rule(self, rule_id: str):
        """启用规则"""
        rule = self.get_rule(rule_id)
        if not rule:
            raise ValueError(f"Rule {rule_id} not found")
        rule.enabled = True

    def disable_rule(self, rule_id: str):
        """禁用规则"""
        rule = self.get_rule(rule_id)
        if not rule:
            raise ValueError(f"Rule {rule_id} not found")
        rule.enabled = False

    def _convert_old_rule(self, rule_data: Dict[str, Any]) -> CombinedDispatchRule:
        """将旧版本规则数据转换为新版本格式"""
        rule_type = rule_data["type"]
        rule_class = DispatchRule.get_rule_type(rule_type)

        # 提取规则配置
        config_fields = rule_class.config_class.model_fields.keys()
        rule_config = {k: rule_data[k] for k in config_fields if k in rule_data}

        # 创建简单规则
        simple_rule = SimpleDispatchRule(type=rule_type, config=rule_config)

        # 创建规则组
        rule_group = RuleGroup(operator="or", rules=[simple_rule])

        # 创建组合规则
        return CombinedDispatchRule(
            rule_id=rule_data["rule_id"],
            name=rule_data["name"],
            description=rule_data.get("description", ""),
            workflow_id=rule_data["workflow_id"],
            rule_groups=[rule_group],
            priority=rule_data.get("priority", 5),
            enabled=rule_data.get("enabled", True),
            metadata=rule_data.get("metadata", {}),
        )

    def load_rules(self, rules_dir: Optional[str] = None):
        """从指定目录加载所有调度规则"""
        rules_dir = rules_dir or self.rules_dir
        if not os.path.exists(rules_dir):
            os.makedirs(rules_dir)

        yaml = YAML(typ="safe")

        for file_name in os.listdir(rules_dir):
            if not file_name.endswith(".yaml"):
                continue

            file_path = os.path.join(rules_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    rules_data = yaml.load(f)

                if not isinstance(rules_data, list):
                    self.logger.warning(
                        f"Invalid rules file {file_name}, expected list of rules"
                    )
                    continue

                for rule_data in rules_data:
                    try:
                        # 检查是否是新版本的组合规则
                        if "rule_groups" in rule_data:
                            rule = CombinedDispatchRule(**rule_data)
                        else:
                            # 旧版本规则，转换为新格式
                            rule = self._convert_old_rule(rule_data)

                        self.register(rule)
                        self.logger.debug(f"Loaded rule: {rule}")
                    except Exception as e:
                        self.logger.error(
                            f"Failed to load rule in file {file_path}: {str(e)}"
                        )

            except Exception as e:
                self.logger.error(f"Failed to load rules from {file_path}: {str(e)}")

    def save_rules(self, rules_dir: Optional[str] = None):
        """保存所有规则到文件"""
        rules_dir = rules_dir or self.rules_dir
        if not os.path.exists(rules_dir):
            os.makedirs(rules_dir)

        yaml = YAML()
        yaml.default_flow_style = False

        # 保存规则
        rules_data = [rule.dict() for rule in self.rules.values()]

        # 保存到文件
        file_path = os.path.join(rules_dir, "rules.yaml")
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(rules_data, f)

# 注册所有规则类型
DispatchRule.register_rule_type(RegexMatchRule)
DispatchRule.register_rule_type(PrefixMatchRule)
DispatchRule.register_rule_type(KeywordMatchRule)
DispatchRule.register_rule_type(BotMentionMatchRule)
DispatchRule.register_rule_type(RandomChanceMatchRule)
DispatchRule.register_rule_type(ChatSenderMatchRule)
DispatchRule.register_rule_type(ChatSenderMismatchRule)
DispatchRule.register_rule_type(ChatTypeMatchRule)
DispatchRule.register_rule_type(IMInstanceMatchRule)
DispatchRule.register_rule_type(FallbackMatchRule)
