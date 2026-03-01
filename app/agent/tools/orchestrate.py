import builtins
import json
import math
import uuid
from datetime import datetime
from typing import Annotated, Any, Union

from pydantic import BaseModel, Discriminator, Field, Tag
from pygents import ContextItem, ContextPool, ContextQueue, Turn, tool

from app.agent.integrations.calendar_service import CalendarEvent, CalendarService
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
    "calendar.read": {
        "description": "Read all calendar events.",
        "kwargs": {},
        "output_schema": "List[dict] — event_id (str), title (str), start_time (ISO 8601 str), end_time (ISO 8601 str), description (str).",
    },
    "calendar.create": {
        "description": "Create a calendar event.",
        "kwargs": {
            "title": {"required": True, "type": "str"},
            "start_time": {"required": True, "type": "ISO 8601 str"},
            "end_time": {"required": True, "type": "ISO 8601 str"},
            "description": {"required": False, "default": ""},
        },
        "output_schema": "str — confirmation message.",
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


async def _add_final_result(result: Any, store_as: str, pool: ContextPool) -> None:
    if result is not None:
        item_id = f"tool_{store_as}_{uuid.uuid4().hex[:8]}"
        await pool.add(
            ContextItem(
                content=result,
                description=store_as,
                id=item_id,
            )
        )


async def _run_pipeline(steps: list, tools: dict, pool: ContextPool) -> None:
    ctx: dict = {}
    last = len(steps) - 1
    for index, step in enumerate(steps):
        if isinstance(step, ToolStep):
            ctx[step.store_as] = await tools[step.name](**step.kwargs)
            if index == last:
                await _add_final_result(ctx[step.store_as], step.store_as, pool)
        elif isinstance(step, TransformStep):
            exec(step.code, {**SAFE_NS, **ctx, "ctx": ctx}, ctx)
            if index == last:
                result = ctx.get(step.store_as)
                await _add_final_result(result, step.store_as, pool)


def _make_tools(memory: ContextQueue) -> dict:
    async def _calendar_read():
        events = CalendarService.read_events()
        return [e.model_dump(mode="json") for e in events]

    async def _calendar_create(
        title: str,
        start_time: str,
        end_time: str,
        description: str = "",
    ):
        event = CalendarEvent(
            title=title,
            start_time=datetime.fromisoformat(start_time),
            end_time=datetime.fromisoformat(end_time),
            description=description,
        )
        CalendarService.create_event(event)
        return f"Created event: {title}"

    async def _read_files_wrapper():
        from app.agent.tools.read_files import get_file_contents

        toolkit = get_toolkit()
        contents = await get_file_contents(memory=memory, toolkit=toolkit)
        return contents

    return {
        "calendar.read": _calendar_read,
        "calendar.create": _calendar_create,
        "read_files": _read_files_wrapper,
    }


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
        tools = _make_tools(memory)
        log_orchestration_pipeline(pipeline.model_dump_json(indent=2))
        await _run_pipeline(pipeline.steps, tools, pool)
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
