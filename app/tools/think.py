from typing import Literal

from pydantic import BaseModel, Field
from py_ai_toolkit import PyAIToolkit
from pygents import Turn, Memory, ToolRegistry, tool

from app.factories import get_toolkit
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


async def decide_next_tool(memory: Memory, toolkit: PyAIToolkit) -> str:
    tools_definitions = get_tools_definitions()
    context = "\n".join(str(item) for item in memory)
    result = await toolkit.asend(
        response_model=ToolUse,
        template=THINK_PROMPT,
        tools=tools_definitions,
        context=context,
    )
    return result.content.name


@tool
async def think(memory: Memory):
    toolkit = get_toolkit()
    tool_name = await decide_next_tool(memory=memory, toolkit=toolkit)
    print(f"[Using tool: {tool_name}]")
    target_tool = ToolRegistry.get(tool_name)
    return Turn(target_tool, args=[memory])
