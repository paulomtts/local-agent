"""Memory query and accessor functions for extracting specific views of memory."""

import json
import re
from pathlib import Path

from pygents import ContextPool, ContextQueue

from app.memory.dataclasses import MemoryItem, MemoryItemType, ToolCall, UserMessage

EPISODIC_TIMESTAMP_FORMAT = "%m-%d %H:%M"
EPISODIC_FILE = Path(__file__).resolve().parents[2] / ".memory" / "episodic.md"
WORKING_FILE = Path(__file__).resolve().parents[2] / ".memory" / "working.md"
SEMANTIC_FILE = Path(__file__).resolve().parents[2] / ".memory" / "semantic.md"


def get_recent_context(memory: ContextQueue, n: int = 3) -> str:
    """Get last N items for recency-focused tasks."""
    return "\n".join(str(item.content) for item in memory.items[-n:])


def get_latest_tool_output(memory: ContextQueue) -> str:
    """Get the most recent ToolCall result from memory."""
    for item in reversed(memory.items):
        if isinstance(item.content, ToolCall):
            return str(item.content)
    return "(none)"


def get_user_messages(memory: ContextQueue, n: int = 5) -> str:
    """Extract only user messages for keyword generation."""
    user_items = [
        item for item in memory.items if isinstance(item.content, UserMessage)
    ]
    return "\n".join(str(item.content) for item in user_items[-n:])


def get_working_memory() -> list[MemoryItemType]:
    """Load and parse working memory from working.md file."""
    if not WORKING_FILE.exists():
        return []

    content = WORKING_FILE.read_text().strip()
    if not content:
        return []

    items: list[MemoryItemType] = []
    for section in re.split(r"\n(?=U: |A: |T\[|C: )", content):
        item = MemoryItem.parse(section)
        if item:
            items.append(item)
    return items


def get_episodic_events(n: int = 3) -> str:
    """Get last N episodic events from episodic memory file."""
    if not EPISODIC_FILE.exists():
        return ""

    lines = EPISODIC_FILE.read_text().strip().split("\n")
    events = [line for line in lines if line.startswith("- ")][-n:]
    return "\n".join(events) if events else ""


def get_semantic_facts(n: int | None = None) -> str:
    """Get last N semantic facts from semantic memory file."""
    if not SEMANTIC_FILE.exists():
        return ""

    lines = SEMANTIC_FILE.read_text().strip().split("\n")
    facts = (
        [line for line in lines if line.startswith("- ")][-n:]
        if n
        else [line for line in lines if line.startswith("- ")]
    )
    return "\n".join(facts) if facts else ""


def get_pool_context(pool: ContextPool, context_ids: list[str]) -> str:
    """Get context from pool by context IDs."""
    pool_context = "(none)"
    item = [pool.get(cid) for cid in context_ids]
    if item:
        pool_context = "\n\n".join(
            item.content if isinstance(item.content, str) else json.dumps(item.content)
            for item in item
            if item.description
        )

    return pool_context
