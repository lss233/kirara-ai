from .openai_adapter import OpenAIAdapterChatBase, OpenAIConfig


class DeepSeekConfig(OpenAIConfig):
    api_base: str = "https://api.deepseek.com/v1"


class DeepSeekAdapter(OpenAIAdapterChatBase):
    def __init__(self, config: DeepSeekConfig):
        super().__init__(config)