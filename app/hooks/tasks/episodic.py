from datetime import datetime
from pathlib import Path
from typing import Any

from py_ai_toolkit import PyAIToolkit
from py_ai_toolkit.core.domain.interfaces import LLMConfig
from pydantic import BaseModel, Field

from app.logger import logger

EPISODIC_FILE = Path(__file__).resolve().parents[2] / "memory" / "episodic.md"

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
    events: list[EpisodicEvent] = Field(default_factory=list, description="Extracted episodic events.")


async def extract_episodic_memory(items: list[Any], user_message: str):
    logger.debug("\033[93m[TASK:episodic_memory]\033[0m")
    context = "\n".join(str(item) for item in items)

    config = LLMConfig()
    toolkit = PyAIToolkit(config)
    result = await toolkit.asend(
        response_model=EpisodicExtraction,
        template=EPISODIC_PROMPT,
        context=context,
        user_message=user_message,
    )

    events = result.content.events
    if not events:
        logger.debug("[TASK:episodic_memory] No new events extracted, skipping.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"## {timestamp}", ""] + [f"- {event.content}" for event in events] + [""]
    with EPISODIC_FILE.open("a") as f:
        f.write("\n".join(lines) + "\n")
    logger.debug(f"[TASK:episodic_memory] Appended {len(events)} events.")
