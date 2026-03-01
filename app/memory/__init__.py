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
from .episodic import write_episodic_event
from .hooks import after_append
from .queries import (
    EPISODIC_TIMESTAMP_FORMAT,
    get_episodic_events,
    get_latest_tool_output,
    get_pool_context,
    get_recent_context,
    get_semantic_facts,
    get_user_messages,
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
    "get_latest_tool_output",
    "get_recent_context",
    "get_semantic_facts",
    "get_episodic_events",
    "get_user_messages",
    "get_pool_context",
    "EPISODIC_TIMESTAMP_FORMAT",
    "after_append",
    "get_working_memory",
    "write_episodic_event",
]
