from datetime import date
from pathlib import Path
from typing import Any

from py_ai_toolkit import PyAIToolkit
from py_ai_toolkit.core.domain.interfaces import LLMConfig
from pydantic import BaseModel, Field

from app.core.logger import log_task, log_token_usage

SEMANTIC_FILE = Path(__file__).resolve().parents[3] / ".memory" / "semantic.md"

SEMANTIC_PROMPT = """From the user's latest message, extract stable facts, preferences, relationships, or knowledge worth persisting across sessions.

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

If nothing qualifies, return empty list.

# Recent conversation
{{ recent_conversation }}
"""


class SemanticFact(BaseModel):
    content: str = Field(
        description="A stable, decontextualized fact free of temporal or situational language."
    )


class SemanticExtraction(BaseModel):
    facts: list[SemanticFact] = Field(
        default_factory=list, description="Extracted semantic facts."
    )


def _format_context(items: list[Any], context_limit: int = 5) -> str:
    recent_items = items[-context_limit:]
    return "\n".join(str(item.content) for item in recent_items)


async def _extract_facts_from_llm(recent_conversation: str) -> list[SemanticFact]:
    config = LLMConfig()
    toolkit = PyAIToolkit(config)
    result = await toolkit.asend(
        response_model=SemanticExtraction,
        template=SEMANTIC_PROMPT,
        recent_conversation=recent_conversation,
        today=date.today().isoformat(),
    )

    log_token_usage("semantic", result)
    if not isinstance(result.content, SemanticExtraction):
        raise ValueError(f"Expected SemanticExtraction, got {type(result.content)}")
    return result.content.facts


def _write_semantic_facts(facts: list[SemanticFact]) -> None:
    lines = [f"- {fact.content}" for fact in facts]
    with SEMANTIC_FILE.open("a") as f:
        f.write("\n".join(lines) + "\n")


async def extract_semantic_memory(items: list[Any]):
    """Extract and persist semantic facts from conversation.

    Orchestrates the semantic memory extraction process:
    1. Format recent conversation context
    2. Extract facts via LLM
    3. Write new facts to memory file

    Args:
        items: List of memory items from conversation
        user_message: Latest user message that triggered extraction
    """
    recent_conversation = _format_context(items, context_limit=3)
    facts = await _extract_facts_from_llm(recent_conversation)

    if not facts:
        log_task("semantic_memory", "skip")
        return

    _write_semantic_facts(facts)
    log_task("semantic_memory", f"+{len(facts)} facts")
