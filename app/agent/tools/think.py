from typing import Literal

from py_ai_toolkit import PyAIToolkit
from pydantic import BaseModel, Field
from pygents import Memory, ToolRegistry, Turn, tool

from app.agent.utils.file_search import get_tools_definitions
from app.core.factories import get_toolkit
from app.core.logger import logger, log_prompt
from app.memory import get_recent_context, get_recent_episodic_events


class ToolUse(BaseModel):
    name: Literal["read_files", "respond"] = Field(description="The tool to be used.")


THINK_PROMPT = """You are a helpful assistant with access to tools. Your goal is to support the user by using one of the tools at your disposal. Rules:
- If you don't need external resources, use 'respond' to answer directly
- If you need to read files or gather information, use 'read_files'
- Consider both recent conversation and episodic memory when deciding

# Tools
{{ tools }}

# Working Memory (Recent Items)
{{ working_memory }}

# Episodic Memory (Recent Events)
{{ episodic_events }}
"""


async def decide_next_tool(memory: Memory, toolkit: PyAIToolkit) -> str:
    tools_definitions = get_tools_definitions()
    working_memory = get_recent_context(memory, n=3)
    episodic_events = get_recent_episodic_events(n=5)

    result = await toolkit.asend(
        response_model=ToolUse,
        template=THINK_PROMPT,
        tools=tools_definitions,
        working_memory=working_memory,
        episodic_events=episodic_events or "(No episodic events yet)",
    )

    log_prompt("think", result)
    return result.content.name


@tool
async def think(memory: Memory):
    toolkit = get_toolkit()
    tool_name = await decide_next_tool(memory=memory, toolkit=toolkit)
    logger.debug(f"\033[38;5;208m[TOOL:{tool_name}]\033[0m")
    target_tool = ToolRegistry.get(tool_name)
    return Turn(target_tool, args=[memory])
