from datetime import date
from pathlib import Path
from typing import Any

from py_ai_toolkit import PyAIToolkit
from py_ai_toolkit.core.domain.interfaces import LLMConfig
from pydantic import BaseModel, Field

from app.core.logger import RESET, TASK_TAG, logger, log_prompt

SEMANTIC_FILE = Path(__file__).resolve().parents[3] / ".memory" / "semantic.md"

SEMANTIC_PROMPT = """Extract stable facts, preferences, relationships, or knowledge worth persisting across sessions.

Normalization rules:
1. Remove temporal language ("recently", "yesterday"). State as timeless truth.
   ❌ "User recently bought Nina a bed"
   ✅ "Nina has a bed"
2. Convert relative time to absolute. If "15 years ago" and today is {{ today }}, compute year (~2011).
   ❌ "Nina is about 15 years old"
   ✅ "Nina was born around 2011"
3. Strip conversational context.
   ❌ "User mentioned they have two dogs"
   ✅ "User has two dogs"
4. Atomic facts — one concept per fact. Split compound statements.

Do NOT extract:
- Events/interactions (episodic memory)
- Session-specific information
- Duplicates of existing memory below

If nothing qualifies, return empty list.

# Existing semantic memory
{{ existing }}

# Recent conversation
{{ recent_conversation }}

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


def _read_existing_semantic_memory() -> str:
    if SEMANTIC_FILE.exists():
        return SEMANTIC_FILE.read_text().strip()
    return ""


def _format_context(items: list[Any], context_limit: int = 5) -> str:
    recent_items = items[-context_limit:]
    return "\n".join(str(item) for item in recent_items)


async def _extract_facts_from_llm(
    existing: str, recent_conversation: str, user_message: str
) -> list[SemanticFact]:
    config = LLMConfig()
    toolkit = PyAIToolkit(config)
    result = await toolkit.asend(
        response_model=SemanticExtraction,
        template=SEMANTIC_PROMPT,
        existing=existing,
        recent_conversation=recent_conversation,
        user_message=user_message,
        today=date.today().isoformat(),
    )

    log_prompt("semantic", result)
    return result.content.facts


def _write_semantic_facts(facts: list[SemanticFact]) -> None:
    lines = [f"- {fact.content}" for fact in facts]
    with SEMANTIC_FILE.open("a") as f:
        f.write("\n".join(lines) + "\n")


async def extract_semantic_memory(items: list[Any], user_message: str):
    """Extract and persist semantic facts from conversation.

    Orchestrates the semantic memory extraction process:
    1. Read existing semantic memory
    2. Format recent conversation context
    3. Extract facts via LLM
    4. Write new facts to memory file

    Args:
        items: List of memory items from conversation
        user_message: Latest user message that triggered extraction
    """
    existing = _read_existing_semantic_memory()
    recent_conversation = _format_context(items, context_limit=3)
    facts = await _extract_facts_from_llm(existing, recent_conversation, user_message)

    if not facts:
        logger.debug(f"{TASK_TAG}[TASK:semantic_memory]{RESET} skip")
        return

    _write_semantic_facts(facts)
    logger.debug(f"{TASK_TAG}[TASK:semantic_memory]{RESET} +{len(facts)} facts")
