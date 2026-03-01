"""Dataclass-based memory item types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class MemoryItem(ABC):
    """Base class for all memory items."""
    timestamp: datetime = field(default_factory=datetime.now)

    @abstractmethod
    def to_display_string(self) -> str:
        pass

    def __str__(self) -> str:
        return self.to_display_string()

    @classmethod
    def parse(cls, section: str) -> "MemoryItemType | None":
        """Parse a working.md section string into a typed memory item."""
        section = section.strip()
        if not section:
            return None

        if section.startswith("U: "):
            return UserMessage(content=section[3:].strip())

        if section.startswith("A: "):
            return AssistantResponse(content=section[3:].strip())

        if section.startswith("T["):
            try:
                end_bracket = section.index("]:")
                tool_name = section[2:end_bracket]
                result = section[end_bracket + 2 :].strip()
                return ToolCall(tool_name=tool_name, result=result)
            except ValueError:
                return None

        if section.startswith("C: "):
            return Compaction(summary=section[3:].strip(), items_compacted=0)

        return None


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
