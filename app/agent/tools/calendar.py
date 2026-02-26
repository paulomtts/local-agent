from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from pygents import ContextItem, ContextQueue, Turn, tool

from app.agent.tools.respond import respond
from app.agent.tools.think import think
from app.core.factories import get_toolkit
from app.core.logger import log_token_usage, log_tool_subtool_use
from app.memory import ToolCall, get_recent_context, write_episodic_event
from integrations.calendar_service import CalendarEvent, CalendarService

ROUTE_PROMPT = """Based on the conversation, determine what the user wants to do with their calendar.

# Working Memory (Recent Conversation)
{{ working_memory }}
"""

CREATE_EVENT_PROMPT = """Extract calendar event details from the conversation. Today is {{ today }} and the timezone is {{ timezone }}.
Resolve relative dates (e.g. "tomorrow", "next Monday") to absolute ISO 8601 datetimes.

# Working Memory (Recent Conversation)
{{ working_memory }}

Determine whether you have enough information to create the event.
Required fields: title, start_time, end_time.

If any required field is missing or ambiguous, set ready=False and write a specific question
for the user that includes a concrete suggestion. Ask about one missing field at a time.

If all required fields are present, set ready=True and populate all fields."""


class CalendarAction(BaseModel):
    action: Literal["read", "create"] = Field(
        description="'read' to list events, 'create' to add a new event."
    )


class EventDraft(BaseModel):
    ready: bool = Field(
        description="True only when title, start_time, and end_time are all known."
    )
    question: str | None = Field(
        default=None,
        description=(
            "If not ready: a question asking for the single most important missing detail, "
            "with a concrete suggestion. E.g. 'What time should it start? I'd suggest 10:00 AM.'"
        ),
    )
    title: str | None = Field(default=None, description="Event title, if known.")
    start_time: datetime | None = Field(
        default=None, description="Event start time, if known."
    )
    end_time: datetime | None = Field(
        default=None, description="Event end time, if known."
    )
    description: str = Field(default="", description="Optional event description.")


def _format_events(service: CalendarService) -> str:
    events = service.read_events()
    if not events:
        return "No calendar events found."

    lines = []
    for event in events:
        start = event.start_time.strftime("%Y-%m-%d %H:%M")
        end = event.end_time.strftime("%H:%M")
        lines.append(f"- [{event.event_id}] {event.title} ({start} → {end})")
        if event.description:
            lines.append(f"  {event.description}")
    return "\n".join(lines)


async def _read(memory: ContextQueue):
    formatted = _format_events(CalendarService())
    await memory.append(ContextItem(ToolCall(tool_name="calendar", result=formatted)))
    log_tool_subtool_use("calendar", "read")

    return Turn(think)


async def _create(memory: ContextQueue):
    toolkit = get_toolkit()
    working_memory = get_recent_context(memory, n=5)

    result = await toolkit.asend(
        response_model=EventDraft,
        template=CREATE_EVENT_PROMPT,
        working_memory=working_memory,
        today=datetime.now().isoformat(),
        timezone=datetime.now().astimezone().tzinfo,
    )

    log_tool_subtool_use("calendar", "create")
    log_token_usage("calendar:create", result)
    draft = result.content
    if not isinstance(draft, EventDraft):
        raise ValueError("Expected EventDraft, got %s" % type(draft))

    if not draft.ready or not draft.title or not draft.start_time or not draft.end_time:
        question = draft.question or "Could you provide more details about the event?"
        await memory.append(
            ContextItem(ToolCall(tool_name="calendar", result=question))
        )

        return Turn(respond)

    event = CalendarEvent(
        title=draft.title,
        start_time=draft.start_time,
        end_time=draft.end_time,
        description=draft.description,
    )
    CalendarService().create_event(event)
    write_episodic_event(
        event=f"created calendar event: {event.title}",
        context=event.start_time.strftime("%Y-%m-%d %H:%M"),
    )

    return Turn(respond)


@tool
async def calendar(memory: ContextQueue, action: str | None = None):
    "Use to read or create calendar events."
    if action == "read":
        return await _read(memory)
    if action == "create":
        return await _create(memory)

    # Fallback: route via LLM when think didn't specify a subtool
    toolkit = get_toolkit()
    working_memory = get_recent_context(memory, n=3)

    result = await toolkit.asend(
        response_model=CalendarAction,
        template=ROUTE_PROMPT,
        working_memory=working_memory,
    )

    log_token_usage("calendar", result)
    routed = result.content
    if not isinstance(routed, CalendarAction):
        raise ValueError("Expected CalendarAction, got %s" % type(routed))

    if routed.action == "read":
        return await _read(memory)
    return await _create(memory)
