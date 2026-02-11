from typing import Literal

from pydantic import BaseModel, Field
from py_ai_toolkit import PyAIToolkit
from pygents import ToolRegistry, Turn, Memory

from app.utils import get_tools_definitions


class ToolUse(BaseModel):
    name: Literal["read_files", "respond"] = Field(description="The tool to be used.")
    reasoning: str = Field(description="The reasoning behind your choice.")


THINK_PROMPT = """You are a helpful assistant with access to tools. Your goal is to support the user by using one of the tools at your disposal. Rules:
- If you don't need external resources, us

# Tools
{{ tools }}

# Context
{{ context }}
"""


async def think_tool_impl(memory: Memory, toolkit: PyAIToolkit) -> Turn:
    tools_definitions = get_tools_definitions()
    context = "\n".join(str(item) for item in memory)
    output = await toolkit.asend(
        response_model=ToolUse,
        template=THINK_PROMPT,
        tools=tools_definitions,
        context=context,
    )
    target_tool = ToolRegistry.get(output.content.name)
    return Turn(target_tool, args=[memory])
