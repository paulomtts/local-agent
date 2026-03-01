import uuid
from datetime import datetime

from pydantic import BaseModel, Field
from pygents import ContextItem, ContextQueue, Turn, tool

from app.agent.integrations.calendar_service import CalendarEvent, CalendarService
from app.agent.tools.respond import respond
from app.agent.tools.think import think
from app.core.factories import get_toolkit
from app.core.logger import log_token_usage, log_tool_subtool_use
from app.memory import ToolCall, get_recent_context, write_episodic_event


@tool
async def calendar(action: str | None = None, context_ids: list[str] | None = None):
    "Use to read or create calendar events."
    if action is None or action == "read":
        async for item in read():
            yield item
        return
    if action == "create":
        async for item in create(context_ids=context_ids):
            yield item
        return
    raise ValueError(f"Unknown calendar action: {action!r}")


CREATE_EVENT_PROMPT = """Extract calendar event details from the conversation. Today is {{ current_date }} and the timezone is {{ timezone }}. Rules:

- Resolve relative dates (e.g. "tomorrow", "next Monday") to absolute ISO 8601 datetimes.
- Determine whether you have enough information to create the event.
- If any required field is missing or ambiguous, set ready=False and write a specific question
for the user that includes a concrete suggestion. Ask about one missing field at a time.
- If all required fields are present, set ready=True and populate all fields.

# Working Memory (Recent Conversation)
{{ working_memory }}

# Existing Calendar Events (for conflict checking)
{{ existing_calendar_events }}
"""


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


@calendar.subtool
async def read():
    "Read calendar events."
    log_tool_subtool_use("calendar", "read")
    events = CalendarService.read_events()
    structured = [e.model_dump(mode="json") for e in events]

    item_id = f"cal_read_{uuid.uuid4().hex[:8]}"
    description = (
        (
            "Calendar events. "
            "Schema: List[dict] — event_id (str), title (str), "
            "start_time (ISO 8601 str), end_time (ISO 8601 str), description (str)"
        )
        if structured
        else "No calendar events found."
    )
    yield ContextItem[list](content=structured, description=description, id=item_id)
    yield Turn(think)


@calendar.subtool
async def create(memory: ContextQueue):
    "Create a calendar event."
    toolkit = get_toolkit()
    working_memory = get_recent_context(memory, n=5)
    existing_calendar_events = (
        CalendarService.format_events() or "No calendar events found."
    )

    result = await toolkit.asend(
        response_model=EventDraft,
        template=CREATE_EVENT_PROMPT,
        working_memory=working_memory,
        existing_calendar_events=existing_calendar_events,
        current_date=datetime.now().isoformat(),
        timezone=datetime.now().astimezone().tzinfo,
    )

    log_tool_subtool_use("calendar", "create")
    log_token_usage("calendar:create", result)
    draft = result.content
    if not isinstance(draft, EventDraft):
        raise ValueError("Expected EventDraft, got %s" % type(draft))

    if not draft.ready or not draft.title or not draft.start_time or not draft.end_time:
        question = draft.question or "Could you provide more details about the event?"
        tool_call = ToolCall(tool_name="calendar.create", result=question)
        yield ContextItem[ToolCall](tool_call)
        yield Turn(respond)
        return

    event = CalendarEvent(
        title=draft.title,
        start_time=draft.start_time,
        end_time=draft.end_time,
        description=draft.description,
    )
    CalendarService.create_event(event)
    write_episodic_event(
        event=f"created calendar event: {event.title}",
        context=event.start_time.strftime("%Y-%m-%d %H:%M"),
    )

    item_id = f"cal_create_{uuid.uuid4().hex[:8]}"
    description = (
        f"Created '{draft.title}' at {draft.start_time.strftime('%Y-%m-%d %H:%M')}"
    )
    tool_call = ToolCall(
        tool_name="calendar.create", result=f"[pool:{item_id}] {description}"
    )
    yield ContextItem[ToolCall](tool_call)
    yield Turn(think)
