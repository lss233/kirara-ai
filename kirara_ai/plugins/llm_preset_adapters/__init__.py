from .alibabacloud_adapter import AlibabaCloudAdapter, AlibabaCloudConfig
from .claude_adapter import ClaudeAdapter, ClaudeConfig
from .deepseek_adapter import DeepSeekAdapter, DeepSeekConfig
from .gemini_adapter import GeminiAdapter, GeminiConfig
from .minimax_adapter import MinimaxAdapter, MinimaxConfig
from .moonshot_adapter import MoonshotAdapter, MoonshotConfig
from .ollama_adapter import OllamaAdapter, OllamaConfig
from .openai_adapter import OpenAIAdapter, OpenAIConfig
from .openrouter_adapter import OpenRouterAdapter, OpenRouterConfig
from .siliconflow_adapter import SiliconFlowAdapter, SiliconFlowConfig
from .tencentcloud_adapter import TencentCloudAdapter, TencentCloudConfig
from .volcengine_adapter import VolcengineAdapter, VolcengineConfig
from .mistral_adapter import MistralAdapter, MistralConfig
from .voyage_adapter import VoyageAdapter, VoyageConfig
from .modelscope_adapter import ModelScopeAdapter, ModelScopeConfig

from kirara_ai.logger import get_logger
from kirara_ai.plugin_manager.plugin import Plugin

logger = get_logger("LLMPresetAdapters")


class LLMPresetAdaptersPlugin(Plugin):
    def __init__(self):
        pass

    def on_load(self):
        self.llm_registry.register(
            "OpenAI", OpenAIAdapter, OpenAIConfig
        )
        self.llm_registry.register(
            "DeepSeek", DeepSeekAdapter, DeepSeekConfig
        )
        self.llm_registry.register(
            "Gemini", GeminiAdapter, GeminiConfig
        )
        self.llm_registry.register(
            "Ollama", OllamaAdapter, OllamaConfig
        )
        self.llm_registry.register(
            "Claude", ClaudeAdapter, ClaudeConfig
        )
        self.llm_registry.register(
            "SiliconFlow", SiliconFlowAdapter, SiliconFlowConfig
        )
        self.llm_registry.register(
            "TencentCloud", TencentCloudAdapter, TencentCloudConfig
        )
        self.llm_registry.register(
            "AlibabaCloud", AlibabaCloudAdapter, AlibabaCloudConfig
        )
        self.llm_registry.register(
            "Moonshot", MoonshotAdapter, MoonshotConfig
        )
        self.llm_registry.register(
            "OpenRouter", OpenRouterAdapter, OpenRouterConfig
        )
        self.llm_registry.register(
            "Minimax", MinimaxAdapter, MinimaxConfig
        )
        self.llm_registry.register(
            "Volcengine", VolcengineAdapter, VolcengineConfig
        )
        self.llm_registry.register(
            "Mistral", MistralAdapter, MistralConfig
        )
        self.llm_registry.register(
            "ModelScope", ModelScopeAdapter, ModelScopeConfig
        )
        logger.info("LLMPresetAdaptersPlugin loaded")

    def on_start(self):
        logger.info("LLMPresetAdaptersPlugin started")

    def on_stop(self):
        logger.info("LLMPresetAdaptersPlugin stopped")
