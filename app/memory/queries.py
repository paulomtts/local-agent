"""Memory query and accessor functions for extracting specific views of memory."""

from pathlib import Path

from app.memory.dataclasses import (
    is_assistant_response,
    is_user_message,
)

EPISODIC_TIMESTAMP_FORMAT = "%m-%d %H:%M"
EPISODIC_FILE = Path(__file__).resolve().parents[2] / ".memory" / "episodic.md"


def get_recent_context(memory, n: int = 3) -> str:
    """Get last N items for recency-focused tasks."""
    items = memory.items[-n:]
    return "\n".join(str(item) for item in items)


def get_user_messages_only(memory, n: int = 5) -> str:
    """Extract only user messages for keyword generation."""
    user_items = [item for item in memory.items if is_user_message(item)]
    return "\n".join(str(item) for item in user_items[-n:])


def get_conversation_pairs(memory, n: int = 3) -> str:
    """Get last N user-assistant pairs for response generation."""
    pairs = []
    i = len(memory.items) - 1
    while i >= 0 and len(pairs) < n * 2:
        item = memory.items[i]
        if is_user_message(item) or is_assistant_response(item):
            pairs.insert(0, item)
        i -= 1
    return "\n".join(str(item) for item in pairs)


def get_recent_episodic_events(n: int = 3) -> str:
    """Get last N episodic events from episodic memory file."""
    if not EPISODIC_FILE.exists():
        return ""

    lines = EPISODIC_FILE.read_text().strip().split("\n")
    events = [line for line in lines if line.startswith("- ")][-n:]
    return "\n".join(events) if events else ""
