import uuid
from typing import Optional

from kirara_ai.llm.format.message import LLMChatContentPartType, LLMToolCallContent
from kirara_ai.llm.format.tool import Function, ToolCall


def generate_tool_call_id(name: str) -> str:
    return f"{name}_{str(uuid.uuid4())}"

def pick_tool_calls(calls: list[LLMChatContentPartType]) -> Optional[list[ToolCall]]:
    tool_calls = [
        ToolCall(
            id=call.id,
            function=Function(name=call.name, arguments=call.parameters)
        ) for call in calls if isinstance(call, LLMToolCallContent)
    ]
    if tool_calls:
        return tool_calls
    else:
        return None