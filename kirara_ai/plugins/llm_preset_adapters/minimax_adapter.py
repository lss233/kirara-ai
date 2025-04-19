from .openai_adapter import OpenAIAdapter, OpenAIConfig


class MinimaxConfig(OpenAIConfig):
    api_base: str = "https://api.minimax.chat/v1"

class MinimaxAdapter(OpenAIAdapter):
    def __init__(self, config: MinimaxConfig):
        super().__init__(config)

    def embed(self, req):
        raise NotImplementedError("Minimax does not support embedding")