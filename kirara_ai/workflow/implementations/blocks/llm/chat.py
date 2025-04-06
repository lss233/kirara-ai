import re
from abc import ABC, abstractmethod
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
from kirara_ai.mcp.manager import MCPServerManager
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
        system_prompt = self.substitute_variables(system_prompt_format, executor)
        user_prompt = self.substitute_variables(user_prompt_format, executor)

        content: List[LLMChatContentPartType] = [LLMChatTextContent(text=user_prompt)]
        # 添加图片内容
        for image in user_msg.images or []:
            content.append(LLMChatImageContent(media_id=image.media_id))

        llm_msg = [
            LLMChatMessage(role="system", content=[LLMChatTextContent(text=system_prompt)]),
        ]
        
        if isinstance(memory_content, list) and all(isinstance(item, LLMChatMessage) for item in memory_content):
            llm_msg.extend(memory_content) # type: ignore
            
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
        message_elements: List[MessageElement] = []
        
        for part in resp.message.content:
            if isinstance(part, LLMChatTextContent):
                # 通过 <break> 将回答分为不同的 TextMessage
                for element in part.text.split("<break>"):
                    if element.strip():
                        message_elements.append(TextMessage(element.strip()))
            elif isinstance(part, LLMChatImageContent):
                message_elements.append(ImageMessage(media_id=part.media_id))
        msg = IMMessage(sender=ChatSender.get_bot_sender(), message_elements=message_elements)
        return {"msg": msg}

class ExampleFunction(Block, ABC):
    """
    这个块是抽象function block，没有实际功能，你可以继承这个类，也可以参考这个类自己实现（遵从inputs, outputs格式约定）。
    """
    name = "tool"
    inputs = {
        "im_msg": Input("im_msg", "im 消息", IMMessage, "im 消息", True),
        "tool_call": Input("call_tools", "llm 回应", LLMChatResponse, "接收llm 的函数调用请求，你应该执行函数调用", True),
    }
    outputs = {
        "send_memory": Output("send_memory", "发送记忆模块", list[LLMChatMessage], "你应该在将函数调用期间的llm对话记录存储到记忆模块中"),
        # TODO: 请将所有LLMChatMessage 整合为一个LLMChatRequest。包含tool调用过程中的toolCallContent和toolResultContent。
        # EXAMPLE: 将接收到的call_tools: LLMChatResponse 中LLMChatMessage提取出来，设定role为assistance, 
        # 然后将所有函数调用结果按照LLMChaMessage(role="tool", content=[LLMToolResultContent])，整合为另外一个LLMChatMessage，拼接到上次构建的LLMChatRequest中。
        "request_body": Output("tool_result", "工具回应", LLMChatRequest, "请将全部上下文信息整合为LLMChatRequest")
    }
    container: DependencyContainer

    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> dict[str, Any]:
        return super().execute(**kwargs)
    
class FunctionCalling(Block):
    """
    这个类只负责联系llm, 请将tools变量或者将tool_result变量整合为LLMChatRequest传入。注意同时传入tool_call和tool_result信息。
    注意: 你实现的function block 应该将too_result存入memory中, 本块不会自动存入函数调用期间的llm对话记录.

    具体block信息流转流程图将放置于后续教程中。详情请参见kirara wiki function calling部分。
    """
    name = "function_calling"
    inputs = {
        "msg": Input("msg", "LLM 函数调用请求体", List[LLMChatMessage], "传递一个规范的函数调用请求体"),
    }
    outputs = {
        "resp": Output("resp", "LLM 消息回应", LLMChatResponse, "返回的response, llm认为无需调用tool或者根据tool结果返回"),
        "tool_call": Output("call_tools", "LLM 工具调用请求", LLMChatResponse, "返回的response带有tool_calls字段，你需要根据此字段进行下一个动作")
    }
    container: DependencyContainer
    
    def __init__(self, model_name: Annotated[
            str, 
            # 等待实现： 只列出支持function_calling的模型
            ParamMeta(label="模型 ID, 支持函数调用且不可为空", description="支持函数调用的模型", options_provider=model_name_options_provider)
        ]): 
        self.model_name = model_name
        self.logger = get_logger("FunctionCallingBlock")

    def execute(self, msg: List[LLMChatMessage]) -> Dict[str, Any]: # type: ignore
        if not self.model_name:
            raise ValueError("need a model name which support function calling")
        else:
            self.logger.info(f"Using  model: {self.model_name} to execute function calling")
        llm = self.container.resolve(LLMManager).get_llm(self.model_name)
        if not llm:
            raise ValueError(f"LLM {self.model_name} not found, please check the model name")
        # 在这里指定llm的model
        request_body = LLMChatRequest(messages=msg, model=self.model_name)
        mcp_manager = self.container.resolve(MCPServerManager)
        tools = mcp_manager.get_tools()
        for tool_name, tool in tools.items():
            mcp_tool_info = tool.tool_info
            tool_call = Tool(
                name=tool_name,
                description=mcp_tool_info.description or "",
                parameters=mcp_tool_info.inputSchema,
            )
            if request_body.tools is None:
                request_body.tools = []
            request_body.tools.append(tool_call)

        response: LLMChatResponse = llm.chat(request_body)
        if not response.message.tool_calls:
            self.logger.debug("No tool calls found, return response directly")
            return {"resp": response}
        else:
            self.logger.debug("Tool calls found, return response with tool calls")
            return {"tool_call": response}