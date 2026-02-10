from typing import Literal

from pydantic import BaseModel, Field
from py_ai_toolkit import PyAIToolkit
from pygents import ToolRegistry, Turn

from app.utils import get_tools_definitions


class ToolUse(BaseModel):
    name: Literal["read_files", "respond"] = Field(description="The tool to be used.")
    reasoning: str = Field(description="The reasoning behind your choice.")


THINK_PROMPT = """You are a helpful assistant with access to tools. Your goal is to support the user by using one of the tools at your disposal. Rules:
- If you don't need external resources, us

# Tools
{{ tools }}

# Message
{{ message }}
"""


async def think_tool_impl(message: str, toolkit: PyAIToolkit) -> Turn:
    tools_definitions = get_tools_definitions()
    output = await toolkit.asend(
        response_model=ToolUse,
        template=THINK_PROMPT,
        tools=tools_definitions,
        message=message,
    )
    target_tool = ToolRegistry.get(output.content.name)
    context = f"User message: {message}\n"
    return Turn(target_tool, args=[context])
