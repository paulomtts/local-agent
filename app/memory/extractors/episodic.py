from datetime import datetime
from pathlib import Path

from py_ai_toolkit import PyAIToolkit
from py_ai_toolkit.core.domain.interfaces import LLMConfig
from pydantic import BaseModel, Field

from app.core.logger import log_task, log_token_usage
from app.memory.queries import EPISODIC_TIMESTAMP_FORMAT

EPISODIC_FILE = Path(__file__).resolve().parents[3] / ".memory" / "episodic.md"

EPISODIC_PROMPT = """Extract user actions/intent from the user message. Agent actions are logged deterministically elsewhere.

# RULES
**ATOMIC**: One action per event. Split compound interactions.
**FACT-BASED**: Past tense, include who did what. No narrative fluff.
  ❌ "User shared that they obtained items"
  ✅ "user obtained Diablo IV unique items"

**USER ONLY**: Only extract what the USER did/asked/provided.
**MINIMAL**: Context only if essential.
**NO DUPLICATES**: Skip routine confirmations, duplicates (below), or general facts (semantic memory).
**USER INTENT ONLY**: Extract the user's domain intent (tasks, info they shared, questions about their world). Do NOT extract meta-requests about the agent (e.g. "user asked for JSON/schema", "user requested extraction in format X", "user asked to return X as Y"). Ignore instructions about how the agent should respond or what format to use.
**NOTABLE**: Only extract notable events.
    ❌ "user asked about Nina"
    ✅ "user mentioned he bought a new bed for his dog"

If nothing notable, return empty list.

# EXAMPLES
"I got new Diablo unique items but they don't fit my build"
→ event: "user obtained Diablo IV unique items" | context: "incompatible with build"

"What were we talking about?"
→ event: "user asked about prior conversation"

"Return the extracted events as JSON with this schema: ..."
→ [] (meta-request about agent output format; skip)

---
User: {{ user_message }}
"""


class EpisodicEvent(BaseModel):
    event: str = Field(
        description="The event in past tense. Include who did what. Examples: 'user asked about Nina', 'agent read 3 files', 'user obtained Diablo IV items'"
    )

    context: str | None = Field(
        default=None,
        description="Essential context or outcome (1 short phrase). Examples: 'poison build incompatible', 'found 5 sources'",
    )


class EpisodicExtraction(BaseModel):
    events: list[EpisodicEvent] = Field(
        default_factory=list, description="Extracted episodic events."
    )


async def _extract_events_from_llm(
    user_message: str,
) -> list[EpisodicEvent]:
    config = LLMConfig()
    toolkit = PyAIToolkit(config)
    result = await toolkit.asend(
        response_model=EpisodicExtraction,
        template=EPISODIC_PROMPT,
        user_message=user_message,
    )

    log_token_usage("episodic", result)
    if not isinstance(result.content, EpisodicExtraction):
        raise ValueError(f"Expected EpisodicExtraction, got {type(result.content)}")
    return result.content.events


def _format_episodic_entry(events: list[EpisodicEvent]) -> list[str]:
    """Format events as structured markdown with timestamp."""
    timestamp = datetime.now().strftime(EPISODIC_TIMESTAMP_FORMAT)
    lines = []

    for event in events:
        if event.context:
            line = f"[{timestamp}] {event.event} | {event.context}"
        else:
            line = f"[{timestamp}] {event.event}"
        lines.append(line)

    return lines


def _write_episodic_events(lines: list[str]) -> None:
    with EPISODIC_FILE.open("a") as f:
        f.write("\n".join(lines) + "\n")


def write_episodic_event(
    event: str,
    context: str | None = None,
) -> None:
    """Deterministically log an episodic event without LLM extraction.

    Use this for logging agent actions that are known deterministically
    (tool executions, errors, outcomes) rather than requiring LLM interpretation.

    Args:
        event: What happened in past tense, include who did what (e.g., "agent read 3 files")
        context: Optional essential outcome/detail

    Example:
        log_episodic_event(
            event="agent read 3 files",
            context="config.py, main.py, utils.py"
        )
    """
    episodic_event = EpisodicEvent(
        event=event,
        context=context,
    )
    lines = _format_episodic_entry([episodic_event])
    _write_episodic_events(lines)


async def extract_episodic_memory(user_message: str):
    """Extract and persist user-side episodic events from conversation.

    Extracts only user actions/intent using LLM. Agent actions are logged
    deterministically by tools themselves using log_episodic_event().

    Orchestrates the episodic memory extraction process:
    1. Extract user events via LLM from user message
    2. Format events with timestamp and type
    3. Write events to memory file

    Args:
        user_message: Latest user message that triggered extraction
    """
    events = await _extract_events_from_llm(user_message)
    if not events:
        log_task("episodic_memory", "skip")
        return
    lines = _format_episodic_entry(events)
    _write_episodic_events(lines)
    log_task("episodic_memory", f"+{len(events)} event")
