from datetime import datetime

from pydantic import BaseModel, Field
from pygents import ContextItem, ContextQueue, Turn, tool

from app.core.factories import get_toolkit
from app.core.logger import log_token_usage
from app.memory import ToolCall, get_recent_context, write_episodic_event
from integrations.calendar import CalendarEvent, CalendarService

CREATE_EVENT_PROMPT = """Extract calendar event details from the conversation. Today is {{ today }} and the timezone is {{ timezone }}.
Resolve relative dates (e.g. "tomorrow", "next Monday") to absolute ISO 8601 datetimes.

# Working Memory (Recent Conversation)
{{ working_memory }}

Determine whether you have enough information to create the event.
Required fields: title, start_time, end_time.

If any required field is missing or ambiguous, set ready=False and write a specific question
for the user that includes a concrete suggestion. Ask about one missing field at a time.

If all required fields are present, set ready=True and populate all fields."""


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


@tool
async def create_event(memory: ContextQueue):
    "Use to create a calendar event from the user's request."
    toolkit = get_toolkit()
    working_memory = get_recent_context(memory, n=5)

    result = await toolkit.asend(
        response_model=EventDraft,
        template=CREATE_EVENT_PROMPT,
        working_memory=working_memory,
        today=datetime.now().isoformat(),
        timezone=datetime.now().astimezone().tzinfo,
    )

    log_token_usage("create_event", result)
    draft = result.content
    if not isinstance(draft, EventDraft):
        raise ValueError("Expected EventDraft, got %s" % type(draft))

    if not draft.ready or not draft.title or not draft.start_time or not draft.end_time:
        question = draft.question or "Could you provide more details about the event?"
        await memory.append(
            ContextItem(ToolCall(tool_name="create_event", result=question))
        )
        from app.agent.tools.respond import respond

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

    from app.agent.tools.respond import respond

    return Turn(respond)
