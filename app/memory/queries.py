"""Memory query and accessor functions for extracting specific views of memory."""

from pathlib import Path

from pygents import ContextQueue

from app.core.logger import log_hook
from app.memory.dataclasses import MemoryItem, MemoryItemType

EPISODIC_TIMESTAMP_FORMAT = "%m-%d %H:%M"
EPISODIC_FILE = Path(__file__).resolve().parents[2] / ".memory" / "episodic.md"
WORKING_FILE = Path(__file__).resolve().parents[2] / ".memory" / "working.md"


def get_latest_tool_context(memory: ContextQueue) -> str:
    """Get the most recent ToolCall result from memory."""
    for item in reversed(memory.items):
        if item.content.is_tool_call():
            return str(item.content)
    return "(none)"


def get_recent_context(memory: ContextQueue, n: int = 3) -> str:
    """Get last N items for recency-focused tasks."""
    return "\n".join(str(item.content) for item in memory.items[-n:])


def get_user_messages_only(memory: ContextQueue, n: int = 5) -> str:
    """Extract only user messages for keyword generation."""
    user_items = [item for item in memory.items if item.content.is_user_message()]
    return "\n".join(str(item.content) for item in user_items[-n:])


def get_recent_episodic_events(n: int = 3) -> str:
    """Get last N episodic events from episodic memory file."""
    if not EPISODIC_FILE.exists():
        return ""

    lines = EPISODIC_FILE.read_text().strip().split("\n")
    events = [line for line in lines if line.startswith("- ")][-n:]
    return "\n".join(events) if events else ""


def get_working_memory() -> list[MemoryItemType]:
    """Load and parse working memory from working.md file."""
    if not WORKING_FILE.exists():
        log_hook("working", "load", "fresh")
        return []

    content = WORKING_FILE.read_text().strip()
    if not content:
        log_hook("working", "load", "fresh")
        return []

    items: list[MemoryItemType] = []
    for section in content.split("\n\n---\n\n"):
        item = MemoryItem.parse(section)
        if item:
            items.append(item)

    log_hook("working", "load", f"{len(items)} items")
    return items
