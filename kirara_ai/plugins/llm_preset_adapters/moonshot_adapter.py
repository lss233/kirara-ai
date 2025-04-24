from .openai_adapter import OpenAIAdapterChatBase, OpenAIConfig

# https://platform.moonshot.cn/docs/intro#%E6%96%87%E6%9C%AC%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B
# TODO: implement feature usages

class MoonshotConfig(OpenAIConfig):
    api_base: str = "https://api.moonshot.cn/v1"


class MoonshotAdapter(OpenAIAdapterChatBase):
    def __init__(self, config: MoonshotConfig):
        super().__init__(config)