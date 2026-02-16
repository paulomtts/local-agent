"""Dataclass-based memory item types.

This module defines strongly-typed memory items using Python's built-in dataclasses.
Each item type maintains the same string representation as the legacy format for
LLM compatibility, but provides type safety and extensibility.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, TypeGuard


@dataclass
class MemoryItem(ABC):
    """Base class for all memory items."""
    timestamp: datetime = field(default_factory=datetime.now)

    @abstractmethod
    def to_display_string(self) -> str:
        """Return format for LLM context."""
        pass

    def __str__(self) -> str:
        return self.to_display_string()


@dataclass
class UserMessage(MemoryItem):
    """User input message."""
    content: str = ""
    type: Literal["user"] = field(default="user", init=False)

    def to_display_string(self) -> str:
        return f"U: {self.content}"


@dataclass
class AssistantResponse(MemoryItem):
    """Assistant's response to user."""
    content: str = ""
    type: Literal["assistant"] = field(default="assistant", init=False)

    def to_display_string(self) -> str:
        return f"A: {self.content}"


@dataclass
class ToolCall(MemoryItem):
    """Tool execution result."""
    tool_name: str = ""
    result: str = ""
    success: bool = True
    type: Literal["tool"] = field(default="tool", init=False)

    def to_display_string(self) -> str:
        return f"T[{self.tool_name}]: {self.result}"


@dataclass
class Compaction(MemoryItem):
    """Compacted summary of older conversation."""
    summary: str = ""
    items_compacted: int = 0
    type: Literal["compaction"] = field(default="compaction", init=False)

    def to_display_string(self) -> str:
        return f"C: {self.summary}"


MemoryItemType = UserMessage | AssistantResponse | ToolCall | Compaction


def is_user_message(item: MemoryItemType) -> TypeGuard[UserMessage]:
    """Check if item is a user message."""
    return isinstance(item, UserMessage)


def is_assistant_response(item: MemoryItemType) -> TypeGuard[AssistantResponse]:
    """Check if item is an assistant response."""
    return isinstance(item, AssistantResponse)


def is_tool_call(item: MemoryItemType) -> TypeGuard[ToolCall]:
    """Check if item is a tool call."""
    return isinstance(item, ToolCall)


def is_compaction(item: MemoryItemType) -> TypeGuard[Compaction]:
    """Check if item is a compaction."""
    return isinstance(item, Compaction)
