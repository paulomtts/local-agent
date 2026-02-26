"""Memory module for managing agent memory."""

from pathlib import Path

from .dataclasses import (
    AssistantResponse,
    Compaction,
    MemoryItem,
    MemoryItemType,
    ToolCall,
    UserMessage,
)
from .extractors.episodic import log_episodic_event
from .hooks import after_append
from .queries import (
    EPISODIC_TIMESTAMP_FORMAT,
    get_recent_context,
    get_recent_episodic_events,
    get_user_messages_only,
    get_working_memory,
)

MEMORY_DIR = Path(__file__).resolve().parents[2] / ".memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "MemoryItem",
    "MemoryItemType",
    "UserMessage",
    "AssistantResponse",
    "ToolCall",
    "Compaction",
    "get_recent_context",
    "get_recent_episodic_events",
    "get_user_messages_only",
    "EPISODIC_TIMESTAMP_FORMAT",
    "after_append",
    "get_working_memory",
    "log_episodic_event",
]
