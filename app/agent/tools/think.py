from typing import Any, Literal

from pydantic import BaseModel, Field
from pygents import ContextPool, ContextQueue, ToolRegistry, Turn, tool

from app.core.factories import get_toolkit
from app.core.logger import log_token_usage, log_tool_use
from app.memory import (
    get_episodic_events,
)


class ToolUse(BaseModel):
    name: Literal["read_files", "respond", "calendar", "orchestrate"] = Field(
        description="The tool to be used."
    )
    subtool: str | None = Field(
        default=None,
        description="For tools with subtools, specify the subtool to skip internal routing. E.g. 'create' or 'read' for calendar.",
    )
    context_ids: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of pool items to pass to the next tool. Use for 'respond' so it can include that data in the reply. "
            "Leave empty when calling a tool that fetches data (e.g. calendar read); those tools do not use context_ids."
        ),
    )


THINK_PROMPT = """You are a helpful assistant with access to tools. Your goal is to support the user by using one of the tools at your disposal. Rules:
- If you don't need external resources, choose 'respond' to answer directly.
- If a fetch tool (e.g. calendar read, read_files) has already been called and the Context Pool below lists its result(s), choose 'respond' and pass the relevant pool IDs in context_ids so the reply can use that data. Do not call the same read/fetch tool again.
- Only call a fetch tool when the pool does not yet contain the needed data.


# Working Memory (Recent Items)
{{ working_memory }}

# Episodic Memory (Recent Events)
{{ episodic_events }}

# Tools & Subtools
{{ tools }}

# Context Pool (Available Data)
{{ pool_catalogue }}
"""


async def decide_next_tool(memory: ContextQueue, pool: ContextPool) -> ToolUse:
    toolkit = get_toolkit()
    result = await toolkit.asend(
        response_model=ToolUse,
        template=THINK_PROMPT,
        tools=ToolRegistry.definitions(),
        pool_catalogue=pool.catalogue() or "(no data in pool)",
        working_memory=memory.history(),
        episodic_events=get_episodic_events(n=5) or "(No episodic events yet)",
    )
    log_token_usage("think", result)
    if not isinstance(result.content, ToolUse):
        raise ValueError("Expected ToolUse, got %s" % type(result.content))
    return result.content


@tool
async def think(memory: ContextQueue, pool: ContextPool):
    log_tool_use("think")
    choice = await decide_next_tool(memory=memory, pool=pool)
    tool = ToolRegistry.get(choice.name)
    kwargs: dict[str, Any] = dict(context_ids=choice.context_ids)
    if choice.subtool:
        kwargs["action"] = choice.subtool
    return Turn(tool, kwargs=kwargs)
