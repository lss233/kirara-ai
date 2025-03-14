import re
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional

from kirara_ai.im.message import IMMessage, TextMessage
from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.llm.format import LLMChatMessage, LLMChatTextContent
from kirara_ai.llm.format.message import LLMChatImageContent
from kirara_ai.llm.format.request import LLMChatRequest
from kirara_ai.llm.format.response import LLMChatResponse
from kirara_ai.llm.llm_manager import LLMManager
from kirara_ai.llm.llm_registry import LLMAbility
from kirara_ai.logger import get_logger
from kirara_ai.workflow.core.block import Block, Input, Output, ParamMeta
from kirara_ai.workflow.core.execution.executor import WorkflowExecutor


def model_name_options_provider(container: DependencyContainer, block: Block) -> List[str]:
    llm_manager: LLMManager = container.resolve(LLMManager)
    return llm_manager.get_supported_models(LLMAbility.TextChat)

class ChatMessageConstructor(Block):
    name = "chat_message_constructor"
    inputs = {
        "user_msg": Input("user_msg", "本轮消息", IMMessage, "用户消息"),
        "user_prompt_format": Input(
            "user_prompt_format", "本轮消息格式", str, "本轮消息格式", default=""
        ),
        "memory_content": Input("memory_content", "上下文消息", str, "历史消息对话"),
        "system_prompt_format": Input(
            "system_prompt_format", "上下文消息格式", str, "上下文消息格式", default=""
        ),
    }
    outputs = {
        "llm_msg": Output(
            "llm_msg", "LLM 对话记录", List[LLMChatMessage], "LLM 对话记录"
        )
    }
    container: DependencyContainer

    def substitute_variables(self, text: str, executor: WorkflowExecutor) -> str:
        """
        替换文本中的变量占位符，支持对象属性和字典键的访问

        :param text: 包含变量占位符的文本，格式为 {variable_name} 或 {variable_name.attribute}
        :param executor: 工作流执行器实例
        :return: 替换后的文本
        """

        def replace_var(match):
            var_path = match.group(1).split(".")
            var_name = var_path[0]

            # 获取基础变量
            value = executor.get_variable(var_name, match.group(0))

            # 如果有属性/键访问
            for attr in var_path[1:]:
                try:
                    # 尝试字典键访问
                    if isinstance(value, dict):
                        value = value.get(attr, match.group(0))
                    # 尝试对象属性访问
                    elif hasattr(value, attr):
                        value = getattr(value, attr)
                    else:
                        # 如果无法访问，返回原始占位符
                        return match.group(0)
                except Exception:
                    # 任何异常都返回原始占位符
                    return match.group(0)

            return str(value)

        return re.sub(r"\{([^}]+)\}", replace_var, text)

    def execute(
        self,
        user_msg: IMMessage,
        memory_content: str,
        system_prompt_format: str = "",
        user_prompt_format: str = "",
    ) -> Dict[str, Any]:
        # 获取当前执行器
        executor = self.container.resolve(WorkflowExecutor)

        # 先替换自有的两个变量
        system_prompt_format = system_prompt_format.replace(
            "{current_date_time}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        system_prompt_format = system_prompt_format.replace(
            "{user_msg}", user_msg.content
        )
        system_prompt_format = system_prompt_format.replace(
            "{user_name}", user_msg.sender.display_name
        )
        system_prompt_format = system_prompt_format.replace(
            "{user_id}", user_msg.sender.user_id
        )
        system_prompt_format = system_prompt_format.replace(
            "{memory_content}", memory_content
        )

        user_prompt_format = user_prompt_format.replace("{user_msg}", user_msg.content)
        user_prompt_format = user_prompt_format.replace(
            "{user_name}", user_msg.sender.display_name
        )
        user_prompt_format = user_prompt_format.replace(
            "{user_id}", user_msg.sender.user_id
        )
        user_prompt_format = user_prompt_format.replace(
            "{memory_content}", memory_content
        )

        # 再替换其他变量
        system_prompt = self.substitute_variables(system_prompt_format, executor)
        user_prompt = self.substitute_variables(user_prompt_format, executor)

        content = [LLMChatTextContent(text=user_prompt)]
        # 添加图片内容
        for image in user_msg.images or []:
            content.append(LLMChatImageContent(media_id=image.media_id))

        llm_msg = [
            LLMChatMessage(role="system", content=[LLMChatTextContent(text=system_prompt)]),
            LLMChatMessage(role="user", content=content),
        ]
        return {"llm_msg": llm_msg}


class ChatCompletion(Block):
    name = "chat_completion"
    inputs = {
        "prompt": Input("prompt", "LLM 对话记录", List[LLMChatMessage], "LLM 对话记录")
    }
    outputs = {"resp": Output("resp", "LLM 对话响应", LLMChatResponse, "LLM 对话响应")}
    container: DependencyContainer

    def __init__(
        self,
        model_name: Annotated[
            Optional[str],
            ParamMeta(label="模型 ID", description="要使用的模型 ID", options_provider=model_name_options_provider),
        ] = None,
    ):
        self.model_name = model_name
        self.logger = get_logger("ChatCompletionBlock")

    def execute(self, prompt: List[LLMChatMessage]) -> Dict[str, Any]:
        llm_manager = self.container.resolve(LLMManager)
        model_id = self.model_name
        if not model_id:
            model_id = llm_manager.get_llm_id_by_ability(LLMAbility.TextChat)
            if not model_id:
                raise ValueError("No available LLM models found")
            else:
                self.logger.info(
                    f"Model id unspecified, using default model: {model_id}"
                )
        else:
            self.logger.debug(f"Using specified model: {model_id}")

        llm = llm_manager.get_llm(model_id)
        if not llm:
            raise ValueError(f"LLM {model_id} not found, please check the model name")
        req = LLMChatRequest(messages=prompt, model=model_id)
        return {"resp": llm.chat(req)}


class ChatResponseConverter(Block):
    name = "chat_response_converter"
    inputs = {"resp": Input("resp", "LLM 响应", LLMChatResponse, "LLM 响应")}
    outputs = {"msg": Output("msg", "IM 消息", IMMessage, "IM 消息")}
    container: DependencyContainer

    def execute(self, resp: LLMChatResponse) -> Dict[str, Any]:
        content = ""
        if resp.choices and resp.choices[0].message:
            content = resp.choices[0].message.content

        # 通过 <break> 将回答分为不同的 TextMessage
        message_elements = []
        for element in content.split("<break>"):
            if element.strip():
                message_elements.append(TextMessage(element.strip()))
        msg = IMMessage(sender="<@llm>", message_elements=message_elements)
        return {"msg": msg}
