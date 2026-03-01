import inspect
import json
import re

from pydantic import BaseModel, Field
from pygents import ContextItem, ContextQueue, Turn, tool

from app.agent.tools.calendar import create, read
from app.core.factories import get_toolkit
from app.core.logger import log_orchestration_pipeline, log_token_usage, log_tool_use
from app.memory import (
    ToolCall,
    get_recent_context,
    get_recent_episodic_events,
    write_episodic_event,
)

TOOL_SCHEMAS = {
    "calendar": {
        "description": "Read or create calendar events.",
        "kwargs": {"action": {"required": True, "values": ["read", "create"]}},
        "output": "String: formatted event list (read) or confirmation/question (create).",
    },
    "read_files": {
        "description": "Search codebase for relevant files and return their contents.",
        "kwargs": {},
        "output": "String: file paths and contents.",
    },
}

ORCHESTRATE_PROMPT = """You are orchestrating a multi-step task by calling tools in sequence.

Rules:
- Output plain Python only: async def pipeline(tools): then the body. No markdown, no ```.
- Call tools: result = await tools['name'](kwarg=value)
- No imports. No other functions. No return needed.
- Do NOT call: respond, think, orchestrate — not in tools dict.

# Available Tools
{{ tool_schemas }}

# Working Memory (Recent Conversation)
{{ working_memory }}

# Episodic Memory (Recent Events)
{{ episodic_events }}
"""


class OrchestrateCode(BaseModel):
    reasoning: str = Field(
        description="Step-by-step plan: which tools, in what order, why."
    )
    code: str = Field(
        description=(
            "Plain Python only: async def pipeline(tools): body. "
            "Call tools as: result = await tools['name'](kwarg=value). "
            "No markdown, no code fences, no trailing text or explanation."
        )
    )


def _extract_pipeline_code(raw: str) -> str:
    raw = raw.strip()
    match = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw


def _make_tools(memory: ContextQueue) -> dict:
    def _last_result(before: int) -> str:
        new = memory.items[before:]
        for item in reversed(new):
            if isinstance(item.content, ToolCall):
                return item.content.result
        return ""

    async def _calendar_wrapper(action: str = "read") -> str:
        before = len(memory.items)
        if action == "read":
            await read()
            return _last_result(before)
        elif action == "create":
            await create()
            new_result = _last_result(before)
            if new_result:
                return new_result  # not-ready: question was appended
            # success: _create appended nothing — synthesize confirmation
            await memory.append(
                ContextItem(
                    ToolCall(tool_name="calendar", result="Calendar event created.")
                )
            )
            return "Calendar event created."
        raise ValueError(f"Unknown calendar action: {action!r}")

    async def _read_files_wrapper() -> str:
        from app.agent.tools.read_files import get_file_contents

        toolkit = get_toolkit()
        contents = await get_file_contents(memory=memory, toolkit=toolkit)
        await memory.append(
            ContextItem(ToolCall(tool_name="read_files", result=contents))
        )
        return contents

    return {"calendar": _calendar_wrapper, "read_files": _read_files_wrapper}


async def _run_generated_code(code: str, tools: dict) -> None:
    ns: dict = {}
    exec(compile(code, "<orchestrate>", "exec"), ns)
    pipeline_fn = ns.get("pipeline")
    if pipeline_fn is None or not callable(pipeline_fn):
        raise ValueError("Generated code must define async def pipeline(tools).")
    if not inspect.iscoroutinefunction(pipeline_fn):
        raise ValueError("pipeline must be async.")
    await pipeline_fn(tools)


@tool
async def orchestrate(memory: ContextQueue):
    "Use for multi-step tasks that require calling multiple tools in sequence."
    log_tool_use("orchestrate")

    toolkit = get_toolkit()
    working_memory = get_recent_context(memory, n=5)
    episodic_events = get_recent_episodic_events(n=5) or "(No episodic events yet)"
    tool_schemas = json.dumps(TOOL_SCHEMAS, indent=2)

    result = await toolkit.asend(
        response_model=OrchestrateCode,
        template=ORCHESTRATE_PROMPT,
        tool_schemas=tool_schemas,
        working_memory=working_memory,
        episodic_events=episodic_events,
    )

    log_token_usage("orchestrate", result)
    plan = result.content
    if not isinstance(plan, OrchestrateCode):
        raise ValueError("Expected OrchestrateCode, got %s" % type(plan))

    tools = _make_tools(memory)

    try:
        code = _extract_pipeline_code(plan.code)
        log_orchestration_pipeline(code)
        await _run_generated_code(code, tools)
    except Exception as exc:
        await memory.append(
            ContextItem(
                ToolCall(
                    tool_name="orchestrate",
                    result=f"Orchestration failed: {exc}",
                    success=False,
                )
            )
        )
        write_episodic_event("orchestration pipeline failed", context=str(exc))
        from app.agent.tools.respond import respond

        return Turn(respond)

    write_episodic_event(
        "agent executed orchestration pipeline", context=plan.reasoning[:120]
    )

    from app.agent.tools.respond import respond

    return Turn(respond)
