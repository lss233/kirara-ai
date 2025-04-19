import asyncio
import re
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional

from kirara_ai.im.message import ImageMessage, IMMessage, MessageElement, TextMessage
from kirara_ai.im.sender import ChatSender
from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.llm.format import LLMChatMessage, LLMChatTextContent
from kirara_ai.llm.format.message import LLMChatContentPartType, LLMChatImageContent
from kirara_ai.llm.format.request import LLMChatRequest, Tool
from kirara_ai.llm.format.response import LLMChatResponse
from kirara_ai.llm.llm_manager import LLMManager
from kirara_ai.llm.llm_registry import LLMAbility
from kirara_ai.logger import get_logger
from kirara_ai.memory.composes.base import ComposableMessageType
from kirara_ai.workflow.core.block import Block, Input, Output, ParamMeta
from kirara_ai.workflow.core.execution.executor import WorkflowExecutor


def model_name_options_provider(container: DependencyContainer, block: Block) -> List[str]:
    llm_manager: LLMManager = container.resolve(LLMManager)
    return sorted(llm_manager.get_supported_models(LLMAbility.TextChat))


class ChatMessageConstructor(Block):
    name = "chat_message_constructor"
    inputs = {
        "user_msg": Input("user_msg", "本轮消息", IMMessage, "用户消息"),
        "user_prompt_format": Input(
            "user_prompt_format", "本轮消息格式", str, "本轮消息格式", default=""
        ),
        "memory_content": Input("memory_content", "历史消息对话", List[ComposableMessageType], "历史消息对话"),
        "system_prompt_format": Input(
            "system_prompt_format", "系统提示词", str, "系统提示词", default=""
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
        replacements = {
            "{current_date_time}": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "{user_msg}": user_msg.content,
            "{user_name}": user_msg.sender.display_name,
            "{user_id}": user_msg.sender.user_id
        }

        if isinstance(memory_content, list) and all(isinstance(item, str) for item in memory_content):
            replacements["{memory_content}"] = "\n".join(memory_content)

        for old, new in replacements.items():
            system_prompt_format = system_prompt_format.replace(old, new)
            user_prompt_format = user_prompt_format.replace(old, new)

        # 再替换其他变量
        system_prompt = self.substitute_variables(
            system_prompt_format, executor)
        user_prompt = self.substitute_variables(user_prompt_format, executor)

        content: List[LLMChatContentPartType] = [
            LLMChatTextContent(text=user_prompt)]
        # 添加图片内容
        for image in user_msg.images or []:
            content.append(LLMChatImageContent(media_id=image.media_id))

        llm_msg = [
            LLMChatMessage(role="system", content=[
                           LLMChatTextContent(text=system_prompt)]),
        ]

        if isinstance(memory_content, list) and all(isinstance(item, LLMChatMessage) for item in memory_content):
            llm_msg.extend(memory_content)  # type: ignore

        llm_msg.append(LLMChatMessage(role="user", content=content))
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
            ParamMeta(
                label="模型 ID",
                description="要使用的模型 ID",
                options_provider=model_name_options_provider),
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
            raise ValueError(
                f"LLM {model_id} not found, please check the model name")
        req = LLMChatRequest(messages=prompt, model=model_id)
        return {"resp": llm.chat(req)}


class ChatResponseConverter(Block):
    name = "chat_response_converter"
    inputs = {"resp": Input("resp", "LLM 响应", LLMChatResponse, "LLM 响应")}
    outputs = {"msg": Output("msg", "IM 消息", IMMessage, "IM 消息")}
    container: DependencyContainer

    def execute(self, resp: LLMChatResponse) -> Dict[str, Any]:
        message_elements: List[MessageElement] = []

        for part in resp.message.content:
            if isinstance(part, LLMChatTextContent):
                # 通过 <break> 将回答分为不同的 TextMessage
                for element in part.text.split("<break>"):
                    if element.strip():
                        message_elements.append(TextMessage(element.strip()))
            elif isinstance(part, LLMChatImageContent):
                message_elements.append(ImageMessage(media_id=part.media_id))
        msg = IMMessage(sender=ChatSender.get_bot_sender(),
                        message_elements=message_elements)
        return {"msg": msg}


class ChatCompletionWithTools(Block):
    """
    支持工具调用的LLM对话块
    """
    name = "chat_completion_with_tools"
    inputs = {
        "msg": Input("msg", "LLM 对话记录", List[LLMChatMessage], "LLM 的 prompt，即由 system、user、assistant和工具调用及结果的完整对话记录"),
        "tools": Input("tools", "工具列表", List[Tool], "工具列表")
    }
    outputs = {
        "resp": Output("resp", "LLM 消息回应", LLMChatResponse, "模型返回给用户的消息"),
        "iteration_msgs": Output("iteration_msgs", "迭代消息", List[LLMChatMessage], "迭代过程中产生的所有消息，可以用记忆存储")
    }

    container: DependencyContainer

    def __init__(self, model_name: Annotated[
        str,
        ParamMeta(
            label="模型 ID, 需要支持函数调用",
            description="支持函数调用的模型",
            options_provider=model_name_options_provider)
    ],
        max_iterations: Annotated[
        int,
        ParamMeta(
            label="最大迭代次数",
            description="允许调用模型请求的最大次数，在进行最后一次请求时，模型将不允许调用工具")
    ] = 4):
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.logger = get_logger("Block.ChatCompletionWithTools")

    def execute(self, msg: List[LLMChatMessage], tools: List[Tool]) -> Dict[str, Any]:
        if not self.model_name:
            raise ValueError(
                "need a model name which support function calling")
        else:
            self.logger.info(
                f"Using  model: {self.model_name} to execute function calling")

        loop = self.container.resolve(asyncio.AbstractEventLoop)
        llm = self.container.resolve(LLMManager).get_llm(self.model_name)
        if not llm:
            raise ValueError(
                f"LLM {self.model_name} not found, please check the model name")

        iteration_msgs: List[LLMChatMessage] = []
        iter_count = 0
        while iter_count < self.max_iterations:
            # 在这里指定llm的model
            self.logger.debug(
                f"Iteration {iter_count+1} of {self.max_iterations}")
            request_body = LLMChatRequest(
                messages=msg + iteration_msgs, model=self.model_name)
            if tools is not None and len(tools) > 0:
                request_body.tools = tools

            # 最后一次迭代不调用工具
            if iter_count == self.max_iterations - 1:
                request_body.tool_choice = "none"

            tools_mapping = {t.name: t for t in tools}

            response: LLMChatResponse = llm.chat(request_body)
            iter_count += 1
            iteration_msgs.append(response.message)
            if response.message.tool_calls:
                self.logger.debug("Tool calls found, attempt to invoke tools")
                for tool_call in response.message.tool_calls:
                    actual_tool = tools_mapping.get(tool_call.function.name)
                    if actual_tool:
                        self.logger.debug(
                            f"Invoking tool: {actual_tool.name}({tool_call.function.arguments})")
                        resp_future = asyncio.run_coroutine_threadsafe(
                            actual_tool.invokeFunc(tool_call), loop
                        )
                        tool_result_msg = LLMChatMessage(
                            role="tool", content=[resp_future.result()])
                        iteration_msgs.append(tool_result_msg)
            else:
                self.logger.debug(
                    "No tool calls found, return response directly")
                return {"resp": response, "iteration_msgs": iteration_msgs}
        
        return {"resp": response, "iteration_msgs": iteration_msgs}
