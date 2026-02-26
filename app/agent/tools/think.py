from typing import Literal

from py_ai_toolkit import PyAIToolkit
from pydantic import BaseModel, Field
from pygents import ContextQueue, ToolRegistry, Turn, tool

from app.agent.utils.definitions import get_tools_definitions
from app.core.factories import get_toolkit
from app.core.logger import log_token_usage, logger
from app.memory import (
    get_latest_tool_context,
    get_recent_context,
    get_recent_episodic_events,
)


class ToolUse(BaseModel):
    name: Literal["read_files", "respond", "calendar"] = Field(
        description="The tool to be used."
    )
    subtool: str | None = Field(
        default=None,
        description="For tools with subtools, specify the subtool to skip internal routing. E.g. 'create' or 'read' for calendar.",
    )


THINK_PROMPT = """You are a helpful assistant with access to tools. Your goal is to support the user by using one of the tools at your disposal. Rules:
- If you don't need external resources, use 'respond' to answer directly
- Consider both recent conversation and episodic memory when deciding
- When talking about data from external resources, prioritize using tools to fetch the data rather than responding directly from what is available in the working memory

# Tools
{{ tools }}

# Working Memory (Recent Items)
{{ working_memory }}

# Episodic Memory (Recent Events)
{{ episodic_events }}

# Tool Context (Last Tool Result)
{{ tool_context_status }}
"""


async def decide_next_tool(memory: ContextQueue, toolkit: PyAIToolkit) -> ToolUse:
    tools_definitions = get_tools_definitions()
    working_memory = get_recent_context(memory, n=3)
    episodic_events = get_recent_episodic_events(n=5)
    tool_context_status = get_latest_tool_context(memory)

    result = await toolkit.asend(
        response_model=ToolUse,
        template=THINK_PROMPT,
        tools=tools_definitions,
        working_memory=working_memory,
        episodic_events=episodic_events or "(No episodic events yet)",
        tool_context_status=tool_context_status,
    )

    log_token_usage("think", result)
    if not isinstance(result.content, ToolUse):
        raise ValueError("Expected ToolUse, got %s" % type(result.content))
    return result.content


@tool
async def think(memory: ContextQueue):
    toolkit = get_toolkit()
    tool_use = await decide_next_tool(memory=memory, toolkit=toolkit)
    logger.debug(f"\033[38;5;208m[TOOL:{tool_use.name}]\033[0m")

    if tool_use.name == "respond":
        from app.agent.tools.respond import respond

        return Turn(respond)

    target_tool = ToolRegistry.get(tool_use.name)
    if tool_use.subtool:
        return Turn(target_tool, kwargs={"action": tool_use.subtool})
    return Turn(target_tool)
