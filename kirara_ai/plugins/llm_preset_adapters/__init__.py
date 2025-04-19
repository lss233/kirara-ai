from llm_preset_adapters.alibabacloud_adapter import AlibabaCloudAdapter, AlibabaCloudConfig
from llm_preset_adapters.claude_adapter import ClaudeAdapter, ClaudeConfig
from llm_preset_adapters.deepseek_adapter import DeepSeekAdapter, DeepSeekConfig
from llm_preset_adapters.gemini_adapter import GeminiAdapter, GeminiConfig
from llm_preset_adapters.minimax_adapter import MinimaxAdapter, MinimaxConfig
from llm_preset_adapters.moonshot_adapter import MoonshotAdapter, MoonshotConfig
from llm_preset_adapters.ollama_adapter import OllamaAdapter, OllamaConfig
from llm_preset_adapters.openai_adapter import OpenAIAdapter, OpenAIConfig
from llm_preset_adapters.openrouter_adapter import OpenRouterAdapter, OpenRouterConfig
from llm_preset_adapters.siliconflow_adapter import SiliconFlowAdapter, SiliconFlowConfig
from llm_preset_adapters.tencentcloud_adapter import TencentCloudAdapter, TencentCloudConfig
from llm_preset_adapters.volcengine_adapter import VolcengineAdapter, VolcengineConfig
from llm_preset_adapters.mistral_adapter import MistralAdapter, MistralConfig
from .voyage_adapter import VoyageAdapter, VoyageConfig

from kirara_ai.llm.llm_registry import LLMAbility
from kirara_ai.logger import get_logger
from kirara_ai.plugin_manager.plugin import Plugin

logger = get_logger("LLMPresetAdapters")


class LLMPresetAdaptersPlugin(Plugin):
    def __init__(self):
        pass

    def on_load(self):
        self.llm_registry.register(
            "OpenAI", OpenAIAdapter, OpenAIConfig, LLMAbility.TextChat
        )
        self.llm_registry.register(
            "DeepSeek", DeepSeekAdapter, DeepSeekConfig, LLMAbility.TextChat
        )
        self.llm_registry.register(
            "Gemini", GeminiAdapter, GeminiConfig, LLMAbility.TextChat
        )
        self.llm_registry.register(
            "Ollama", OllamaAdapter, OllamaConfig, LLMAbility.TextChat
        )
        self.llm_registry.register(
            "Claude", ClaudeAdapter, ClaudeConfig, LLMAbility.TextChat
        )
        self.llm_registry.register(
            "SiliconFlow", SiliconFlowAdapter, SiliconFlowConfig, LLMAbility.TextChat
        )
        self.llm_registry.register(
            "TencentCloud", TencentCloudAdapter, TencentCloudConfig, LLMAbility.TextChat
        )
        self.llm_registry.register(
            "AlibabaCloud", AlibabaCloudAdapter, AlibabaCloudConfig, LLMAbility.TextChat
        )
        self.llm_registry.register(
            "Moonshot", MoonshotAdapter, MoonshotConfig, LLMAbility.TextChat
        )
        self.llm_registry.register(
            "OpenRouter", OpenRouterAdapter, OpenRouterConfig, LLMAbility.TextChat
        )
        self.llm_registry.register(
            "Minimax", MinimaxAdapter, MinimaxConfig, LLMAbility.TextChat
        )
        self.llm_registry.register(
            "Volcengine", VolcengineAdapter, VolcengineConfig, LLMAbility.TextChat
        )
        self.llm_registry.register(
            "Mistral", MistralAdapter, MistralConfig, LLMAbility.TextChat
        )
        logger.info("LLMPresetAdaptersPlugin loaded")

    def on_start(self):
        logger.info("LLMPresetAdaptersPlugin started")

    def on_stop(self):
        logger.info("LLMPresetAdaptersPlugin stopped")
