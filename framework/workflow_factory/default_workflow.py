import asyncio
from typing import Any, Dict, List
from framework.im.adapter import IMAdapter
from framework.llm.format.message import LLMChatMessage
from framework.llm.format.request import LLMChatRequest
from framework.llm.format.response import LLMChatResponse
from framework.llm.llm_manager import LLMManager
from framework.workflow_executor.workflow import Wire, Workflow
from framework.im.message import IMMessage, TextMessage
from framework.ioc.container import DependencyContainer
from framework.workflow_executor.block import Block
from framework.workflow_executor.input_output import Input, Output

class MessageInput(Block):
    def __init__(self, container: DependencyContainer):
        outputs = {"msg": Output("msg", IMMessage, "Input message")}
        super().__init__("msg_input", {}, outputs)
        self.container = container

    def execute(self, **kwargs) -> Dict[str, Any]:
        msg = self.container.resolve(IMMessage)
        return {"msg": msg}

class MessageToLLM(Block):
    def __init__(self, container: DependencyContainer):
        inputs = {"msg": Input("msg", IMMessage, "Input message")}
        outputs = {"llm_msg": Output("llm_msg", List[LLMChatMessage], "LLM message")}
        super().__init__("msg_to_llm", inputs, outputs)
        self.container = container

    def execute(self, **kwargs) -> Dict[str, Any]:
        msg = kwargs["msg"]
        llm_msg = LLMChatMessage(role='user', content=msg.content)
        return {"llm_msg": [llm_msg]}

class LLMChat(Block):
    def __init__(self, container: DependencyContainer):
        inputs = {"prompt": Input("prompt", List[LLMChatMessage], "LLM prompt")}
        outputs = {"resp": Output("resp", LLMChatResponse, "LLM response")}
        super().__init__("llm_chat", inputs, outputs)
        self.container = container

    def execute(self, **kwargs) -> Dict[str, Any]:
        prompt = kwargs["prompt"]
        llm = self.container.resolve(LLMManager).get_llm('deepseek-r1')
        req = LLMChatRequest(messages=prompt, model='DeepSeek-R1')
        return {"resp": llm.chat(req)}

class LLMToMessage(Block):
    def __init__(self, container: DependencyContainer):
        inputs = {"resp": Input("resp", LLMChatResponse, "LLM response")}
        outputs = {"msg": Output("msg", IMMessage, "Output message")}
        super().__init__("llm_to_msg", inputs, outputs)
        self.container = container

    def execute(self, **kwargs) -> Dict[str, Any]:
        resp = kwargs["resp"]
        content = ""
        if resp.choices and resp.choices[0].message:
            content = resp.choices[0].message.content
            
        msg = IMMessage(
            sender="<@llm>",
            message_elements=[TextMessage(content)]
        )
        return {"msg": msg}

class MessageSender(Block):
    def __init__(self, container: DependencyContainer):
        inputs = {"msg": Input("msg", IMMessage, "Message to send")}
        super().__init__("msg_sender", inputs, {})
        self.container = container

    def execute(self, **kwargs) -> Dict[str, Any]:
        msg = kwargs["msg"]
        src_msg = self.container.resolve(IMMessage)
        adapter = self.container.resolve(IMAdapter)
        loop: asyncio.AbstractEventLoop = self.container.resolve(asyncio.AbstractEventLoop)
        loop.create_task(adapter.send_message(msg, src_msg.sender))
        # return {"ok": True}

class DefaultWorkflow(Workflow):
    def __init__(self, container: DependencyContainer):
        msg_input = MessageInput(container)
        msg_to_llm = MessageToLLM(container) 
        llm_chat = LLMChat(container)
        llm_to_msg = LLMToMessage(container)
        msg_sender = MessageSender(container)

        wires = [
            Wire(msg_input, "msg", msg_to_llm, "msg"),
            Wire(msg_to_llm, "llm_msg", llm_chat, "prompt"),
            Wire(llm_chat, "resp", llm_to_msg, "resp"),
            Wire(llm_to_msg, "msg", msg_sender, "msg")
        ]

        super().__init__(
            [msg_input, msg_to_llm, llm_chat, llm_to_msg, msg_sender],
            wires
        )
        
        self.container = container