from datetime import datetime
from pathlib import Path
from typing import Any

from py_ai_toolkit import PyAIToolkit
from py_ai_toolkit.core.domain.interfaces import LLMConfig
from pydantic import BaseModel, Field

from app.core.logger import logger
from app.memory.queries import EPISODIC_TIMESTAMP_FORMAT

EPISODIC_FILE = Path(__file__).resolve().parents[3] / ".memory" / "episodic.md"

EPISODIC_PROMPT = """You are maintaining an episodic memory — a log of personal experiences tied to a specific moment in time. Episodic memory answers: "What happened, and in what context?"

From the latest user message and conversation, extract interaction traces worth remembering: tasks attempted, problems encountered, decisions made, tools used and their outcomes, errors hit and how they were resolved, or any other concrete event from this session. Each entry should read like a brief diary note — grounded in what specifically happened, not abstract knowledge.

Do NOT extract:
- General facts, preferences, or definitions (those belong in semantic memory)
- Anything that isn't tied to a concrete interaction or event

If nothing notable happened, return an empty list.

# Conversation context
{{ context }}

# Latest user message
{{ user_message }}
"""


class EpisodicEvent(BaseModel):
    content: str = Field(description="A notable event, interaction, or outcome.")


class EpisodicExtraction(BaseModel):
    events: list[EpisodicEvent] = Field(
        default_factory=list, description="Extracted episodic events."
    )


def _format_context(items: list[Any], context_limit: int = 5) -> str:
    recent_items = items[-context_limit:]
    return "\n".join(str(item) for item in recent_items)


async def _extract_events_from_llm(
    context: str, user_message: str
) -> list[EpisodicEvent]:
    config = LLMConfig()
    toolkit = PyAIToolkit(config)
    result = await toolkit.asend(
        response_model=EpisodicExtraction,
        template=EPISODIC_PROMPT,
        context=context,
        user_message=user_message,
    )
    return result.content.events


def _format_episodic_entry(events: list[EpisodicEvent]) -> list[str]:
    timestamp = datetime.now().strftime(EPISODIC_TIMESTAMP_FORMAT)
    lines = [f"## {timestamp}", ""]
    lines.extend(f"- {event.content}" for event in events)
    lines.append("")
    return lines


def _write_episodic_events(lines: list[str]) -> None:
    with EPISODIC_FILE.open("a") as f:
        f.write("\n".join(lines) + "\n")


async def extract_episodic_memory(items: list[Any], user_message: str):
    """Extract and persist episodic events from conversation.

    Orchestrates the episodic memory extraction process:
    1. Format recent conversation context
    2. Extract events via LLM
    3. Format events with timestamp
    4. Write events to memory file

    Args:
        items: List of memory items from conversation
        user_message: Latest user message that triggered extraction
    """
    logger.debug("\033[93m[TASK:episodic_memory]\033[0m")
    context = _format_context(items, context_limit=5)
    events = await _extract_events_from_llm(context, user_message)
    if not events:
        logger.debug("[TASK:episodic_memory] No new events extracted, skipping.")
        return

    lines = _format_episodic_entry(events)
    _write_episodic_events(lines)
    logger.debug(f"[TASK:episodic_memory] Appended {len(events)} events.")
