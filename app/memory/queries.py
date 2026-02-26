"""Memory query and accessor functions for extracting specific views of memory."""

from pathlib import Path

from pygents import ContextQueue

from app.core.logger import HOOK_TAG, RESET, logger
from app.memory.dataclasses import MemoryItem, MemoryItemType

EPISODIC_TIMESTAMP_FORMAT = "%m-%d %H:%M"
EPISODIC_FILE = Path(__file__).resolve().parents[2] / ".memory" / "episodic.md"
WORKING_FILE = Path(__file__).resolve().parents[2] / ".memory" / "working.md"


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
        logger.debug(
            f"{HOOK_TAG}[HOOK:load_working]{RESET} file not found, starting fresh"
        )
        return []

    content = WORKING_FILE.read_text().strip()
    if not content:
        logger.debug(f"{HOOK_TAG}[HOOK:load_working]{RESET} empty file, starting fresh")
        return []

    items: list[MemoryItemType] = []
    for section in content.split("\n\n---\n\n"):
        item = MemoryItem.parse(section)
        if item:
            items.append(item)

    logger.debug(f"{HOOK_TAG}[HOOK:load_working]{RESET} loaded {len(items)} items")
    return items
