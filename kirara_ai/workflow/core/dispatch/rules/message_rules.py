import re
from typing import List

from pydantic import Field

from kirara_ai.im.message import IMMessage, MentionElement
from kirara_ai.im.sender import ChatSender
from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.workflow.core.workflow.registry import WorkflowRegistry

from .base import DispatchRule, RuleConfig


class RegexRuleConfig(RuleConfig):
    """正则规则配置"""
    pattern: str = Field(title="正则表达式", description="正则表达式")

class RegexMatchRule(DispatchRule):
    """根据正则表达式匹配的规则"""
    config_class = RegexRuleConfig
    type_name = "regex"

    def __init__(self, pattern: str, workflow_registry: WorkflowRegistry, workflow_id: str):
        super().__init__(workflow_registry, workflow_id)
        self.pattern = re.compile(pattern)

    def match(self, message: IMMessage, container: DependencyContainer) -> bool:
        return bool(self.pattern.search(message.content))

    def get_config(self) -> RegexRuleConfig:
        return RegexRuleConfig(pattern=self.pattern.pattern)

    @classmethod
    def from_config(cls, config: RegexRuleConfig, workflow_registry: WorkflowRegistry, workflow_id: str) -> "RegexMatchRule":
        return cls(config.pattern, workflow_registry, workflow_id)

class PrefixRuleConfig(RuleConfig):
    """前缀规则配置"""
    prefix: str = Field(title="前缀", description="前缀")

class PrefixMatchRule(DispatchRule):
    """根据消息前缀匹配的规则"""
    config_class = PrefixRuleConfig
    type_name = "prefix"

    def __init__(self, prefix: str, workflow_registry: WorkflowRegistry, workflow_id: str):
        super().__init__(workflow_registry, workflow_id)
        self.prefix = prefix

    def match(self, message: IMMessage, container: DependencyContainer) -> bool:
        return message.content.startswith(self.prefix)

    def get_config(self) -> PrefixRuleConfig:
        return PrefixRuleConfig(prefix=self.prefix)

    @classmethod
    def from_config(cls, config: PrefixRuleConfig, workflow_registry: WorkflowRegistry, workflow_id: str) -> "PrefixMatchRule":
        return cls(config.prefix, workflow_registry, workflow_id)


class KeywordRuleConfig(RuleConfig):
    """关键词规则配置"""
    keywords: List[str] = Field(title="关键词", description="关键词列表")
    
class KeywordMatchRule(DispatchRule):
    """根据关键词匹配的规则"""
    config_class = KeywordRuleConfig
    type_name = "keyword"

    def __init__(self, keywords: list[str], workflow_registry: WorkflowRegistry, workflow_id: str):
        super().__init__(workflow_registry, workflow_id)
        self.keywords = keywords

    def match(self, message: IMMessage, container: DependencyContainer) -> bool:
        return any(keyword in message.content for keyword in self.keywords)

    def get_config(self) -> KeywordRuleConfig:
        return KeywordRuleConfig(keywords=self.keywords)

    @classmethod
    def from_config(cls, config: KeywordRuleConfig, workflow_registry: WorkflowRegistry, workflow_id: str) -> "KeywordMatchRule":
        return cls(config.keywords, workflow_registry, workflow_id)
    
class BotMentionMatchRule(DispatchRule):
    """根据机器人被提及匹配的规则"""
    config_class = RuleConfig
    type_name = "bot_mention"

    def __init__(self, workflow_registry: WorkflowRegistry, workflow_id: str):
        super().__init__(workflow_registry, workflow_id)

    def match(self, message: IMMessage, container: DependencyContainer) -> bool:
        bot_sender = ChatSender.get_bot_sender()
        return any(isinstance(element, MentionElement) and element.target == bot_sender for element in message.message_elements)

    def get_config(self) -> RuleConfig: 
        return RuleConfig()

    @classmethod
    def from_config(cls, config: RuleConfig, workflow_registry: WorkflowRegistry, workflow_id: str) -> "BotMentionMatchRule":
        return cls(workflow_registry, workflow_id)
