from datetime import date
from pathlib import Path
from typing import Any

from py_ai_toolkit import PyAIToolkit
from py_ai_toolkit.core.domain.interfaces import LLMConfig
from pydantic import BaseModel, Field

from app.core.logger import logger

SEMANTIC_FILE = Path(__file__).resolve().parents[3] / ".memory" / "semantic.md"

SEMANTIC_PROMPT = """You are maintaining a semantic memory — a structured store of durable, decontextualized knowledge. Semantic memory stores what is true, not what happened or when it was learned.

From the latest user message and conversation, extract stable facts, preferences, relationships, or knowledge worth persisting across sessions.

Normalization rules — apply these strictly:
1. Remove all temporal language ("recently", "just", "yesterday", "a while ago"). State the fact as a timeless truth.
   BAD:  "User recently bought Nina a new bed and she loved it."
   GOOD: "Nina has a bed that she likes."
2. Convert relative time references to absolute values when possible. If the user says "about 15 years ago" and today is {{ today }}, compute the year (e.g. ~2011). If an age is mentioned, store birth year instead.
   BAD:  "Nina is about 15 years old."
   GOOD: "Nina was born around 2011."
3. Strip conversational context. Do not mention that something was "said", "mentioned", or "discussed".
   BAD:  "User mentioned they have two dogs."
   GOOD: "User has two dogs."
4. Facts should be atomic — one concept per fact. Split compound statements.
5. If a new fact contradicts or updates an existing one, extract the updated version. It will replace the old one.

Do NOT extract:
- Events, interactions, or outcomes (those belong in episodic memory)
- Session-specific or situational information
- Facts already present and unchanged in the existing memory below

If nothing qualifies, return an empty list.

# Existing semantic memory
{{ existing }}

# Conversation context
{{ context }}

# Latest user message
{{ user_message }}
"""


class SemanticFact(BaseModel):
    content: str = Field(
        description="A stable, decontextualized fact free of temporal or situational language."
    )


class SemanticExtraction(BaseModel):
    facts: list[SemanticFact] = Field(
        default_factory=list, description="Extracted semantic facts."
    )


async def extract_semantic_memory(items: list[Any], user_message: str):
    logger.debug("\033[93m[TASK:semantic_memory]\033[0m")
    existing = SEMANTIC_FILE.read_text().strip() if SEMANTIC_FILE.exists() else ""
    # Use only last 5 items for context instead of all items
    context = "\n".join(str(item) for item in items[-5:])

    config = LLMConfig()
    toolkit = PyAIToolkit(config)
    result = await toolkit.asend(
        response_model=SemanticExtraction,
        template=SEMANTIC_PROMPT,
        existing=existing,
        context=context,
        user_message=user_message,
        today=date.today().isoformat(),
    )

    facts = result.content.facts
    if not facts:
        logger.debug("[TASK:semantic_memory] No new facts extracted, skipping.")
        return

    lines = [f"- {fact.content}" for fact in facts]
    with SEMANTIC_FILE.open("a") as f:
        f.write("\n".join(lines) + "\n")
    logger.debug(f"[TASK:semantic_memory] Appended {len(facts)} facts.")
