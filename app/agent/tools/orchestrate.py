import builtins
import json
import math
from datetime import datetime
from typing import Annotated, Union

from pydantic import BaseModel, Discriminator, Field, Tag
from pygents import ContextItem, ContextPool, ContextQueue, Turn, tool

from app.agent.tools.calendar import create, read
from app.agent.tools.think import think
from app.core.factories import get_toolkit
from app.core.logger import log_orchestration_pipeline, log_token_usage, log_tool_use
from app.memory import (
    ToolCall,
    get_episodic_events,
    get_recent_context,
)
from app.memory.episodic import write_episodic_event

TOOL_SCHEMAS = {
    "calendar": {
        "description": "Read calendar events.",
        "kwargs": {"action": {"required": False, "default": "read"}},
        "output_schema": "List[dict] — event_id (str), title (str), start_time (ISO 8601 str), end_time (ISO 8601 str), description (str)",
    },
    "read_files": {
        "description": "Search codebase for relevant files and return their contents.",
        "kwargs": {},
        "output_schema": "str — file paths and contents.",
    },
}

ORCHESTRATE_PROMPT = """You are orchestrating a multi-step task by calling tools in sequence.

Output a Pipeline with a list of steps. Each step is one of:

**ToolStep** — call a tool:
  - name: tool name (from Available Tools)
  - kwargs: dict of keyword arguments
  - store_as: variable name for the result

**TransformStep** — pure Python logic between tool calls:
  - code: multi-line Python. Reads from ctx variables. Must assign to store_as.
  - store_as: variable name the code assigns to

Available imports in transform code: json, math, datetime.
Do NOT call tools or access memory inside transform code.

# Available Tools
{{ tool_schemas }}

# Working Memory (Recent Conversation)
{{ working_memory }}

# Episodic Memory (Recent Events)
{{ episodic_events }}
"""


class ToolStep(BaseModel):
    """Call a tool and store its result."""

    name: str = Field(description="Tool name.")
    kwargs: dict = Field(default_factory=dict, description="Keyword arguments.")
    store_as: str = Field(description="Variable name to store the result in.")


class TransformStep(BaseModel):
    """Run Python code to transform data between tool calls."""

    code: str = Field(
        description="Multi-line Python. Must assign result to store_as variable."
    )
    store_as: str = Field(description="Variable the code must assign to.")


def _discriminate_step(v) -> str:
    if isinstance(v, dict):
        return "transform" if "code" in v else "tool"
    return "transform" if isinstance(v, TransformStep) else "tool"


PipelineStep = Annotated[
    Union[Annotated[ToolStep, Tag("tool")], Annotated[TransformStep, Tag("transform")]],
    Discriminator(_discriminate_step),
]


class Pipeline(BaseModel):
    reasoning: str = Field(
        description="Step-by-step plan: which tools, in what order, why."
    )
    steps: list[PipelineStep]


SAFE_NS = {
    "__builtins__": builtins,
    "json": json,
    "math": math,
    "datetime": datetime,
}


async def _run_pipeline(steps: list, tools: dict) -> None:
    ctx: dict = {}
    for step in steps:
        if isinstance(step, ToolStep):
            ctx[step.store_as] = await tools[step.name](**step.kwargs)
        elif isinstance(step, TransformStep):
            exec(step.code, {**SAFE_NS, **ctx, "ctx": ctx}, ctx)


def _make_tools(memory: ContextQueue, pool: ContextPool) -> dict:
    async def _calendar_wrapper(action: str = "read"):
        subtool = read if action == "read" else create
        data_item = None
        async for item in subtool():
            if isinstance(item, ContextItem) and item.id and item.description:
                await pool.add(item)
                data_item = item
        return data_item.content if data_item else None

    async def _read_files_wrapper():
        from app.agent.tools.read_files import get_file_contents

        toolkit = get_toolkit()
        contents = await get_file_contents(memory=memory, toolkit=toolkit)
        return contents

    return {"calendar": _calendar_wrapper, "read_files": _read_files_wrapper}


@tool
async def orchestrate(memory: ContextQueue, pool: ContextPool):
    "Use for multi-step tasks that require calling multiple tools in sequence."
    log_tool_use("orchestrate")

    toolkit = get_toolkit()
    result = await toolkit.asend(
        response_model=Pipeline,
        template=ORCHESTRATE_PROMPT,
        tool_schemas=TOOL_SCHEMAS,
        working_memory=get_recent_context(memory, n=5),
        episodic_events=get_episodic_events(n=5) or "(No episodic events yet)",
    )
    log_token_usage("orchestrate", result)

    if not isinstance(result.content, Pipeline):
        raise ValueError("Expected Pipeline, got %s" % type(result.content))

    pipeline = result.content

    try:
        tools = _make_tools(memory, pool)
        log_orchestration_pipeline(pipeline.model_dump_json(indent=2))
        await _run_pipeline(pipeline.steps, tools)
    except Exception as exc:
        tool_call = ToolCall(
            tool_name="orchestrate",
            result=f"Orchestration failed: {exc}",
            success=False,
        )
        yield ContextItem(tool_call)
        yield Turn(think)
        write_episodic_event("orchestration pipeline failed", context=str(exc))
        return

    write_episodic_event(
        "agent executed orchestration pipeline", context=pipeline.reasoning[:120]
    )
    yield Turn(think)
