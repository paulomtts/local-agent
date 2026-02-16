"""Memory module for managing agent memory."""

from pathlib import Path

from .dataclasses import (
    AssistantResponse,
    Compaction,
    MemoryItem,
    MemoryItemType,
    ToolCall,
    UserMessage,
    is_assistant_response,
    is_compaction,
    is_tool_call,
    is_user_message,
)
from .extractors.episodic import log_episodic_event
from .hooks import after_append, load_working_memory
from .queries import (
    EPISODIC_TIMESTAMP_FORMAT,
    get_conversation_pairs,
    get_recent_context,
    get_recent_episodic_events,
    get_user_messages_only,
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
    "is_user_message",
    "is_assistant_response",
    "is_tool_call",
    "is_compaction",
    "get_recent_context",
    "get_recent_episodic_events",
    "get_user_messages_only",
    "get_conversation_pairs",
    "EPISODIC_TIMESTAMP_FORMAT",
    "after_append",
    "load_working_memory",
    "log_episodic_event",
]
