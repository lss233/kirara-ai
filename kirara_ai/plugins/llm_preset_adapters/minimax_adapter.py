from .openai_adapter import OpenAIAdapterChatBase, OpenAIConfig


class MinimaxConfig(OpenAIConfig):
    api_base: str = "https://api.minimax.chat/v1"

class MinimaxAdapter(OpenAIAdapterChatBase):
    def __init__(self, config: MinimaxConfig):
        super().__init__(config)