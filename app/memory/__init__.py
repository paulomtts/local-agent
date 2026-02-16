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
from .hooks import after_append
from .queries import (
    EPISODIC_TIMESTAMP_FORMAT,
    get_conversation_pairs,
    get_recent_context,
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
    "get_user_messages_only",
    "get_conversation_pairs",
    "EPISODIC_TIMESTAMP_FORMAT",
    "after_append",
]
