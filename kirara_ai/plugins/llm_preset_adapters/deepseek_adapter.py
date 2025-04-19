from .openai_adapter import OpenAIAdapter, OpenAIConfig


class DeepSeekConfig(OpenAIConfig):
    api_base: str = "https://api.deepseek.com/v1"


class DeepSeekAdapter(OpenAIAdapter):
    def __init__(self, config: DeepSeekConfig):
        super().__init__(config)

    def embed(self, req):
        raise NotImplementedError("DeepSeek does not support embedding")