"""Tests for memory item dataclasses and type guards."""

from datetime import datetime

from app.memory.dataclasses import (
    AssistantResponse,
    Compaction,
    MemoryItem,
    MemoryItemType,
    ToolCall,
    UserMessage,
)


class TestMemoryItemCreation:
    """Test creating memory items using dataclasses."""

    def test_user_message_creation(self):
        msg = UserMessage(content="Hello world")
        assert isinstance(msg, UserMessage)
        assert msg.content == "Hello world"
        assert msg.type == "user"
        assert str(msg) == "U: Hello world"
        assert msg.to_display_string() == "U: Hello world"

    def test_assistant_response_creation(self):
        resp = AssistantResponse(content="Hi there")
        assert isinstance(resp, AssistantResponse)
        assert resp.content == "Hi there"
        assert resp.type == "assistant"
        assert str(resp) == "A: Hi there"
        assert resp.to_display_string() == "A: Hi there"

    def test_tool_call_creation(self):
        tool = ToolCall(tool_name="read_files", result="Contents...")
        assert isinstance(tool, ToolCall)
        assert tool.tool_name == "read_files"
        assert tool.result == "Contents..."
        assert tool.success is True
        assert tool.type == "tool"
        assert str(tool) == "T[read_files]: Contents..."
        assert tool.to_display_string() == "T[read_files]: Contents..."

    def test_compaction_creation(self):
        comp = Compaction(summary="Previous context", items_compacted=5)
        assert isinstance(comp, Compaction)
        assert comp.summary == "Previous context"
        assert comp.items_compacted == 5
        assert comp.type == "compaction"
        assert str(comp) == "C: Previous context"
        assert comp.to_display_string() == "C: Previous context"

    def test_compaction_default_items_compacted(self):
        comp = Compaction(summary="Summary")
        assert comp.items_compacted == 0

    def test_direct_instantiation(self):
        """Test creating items directly without factory functions."""
        msg = UserMessage(content="Direct message")
        assert msg.content == "Direct message"
        assert msg.type == "user"
        assert str(msg) == "U: Direct message"

        resp = AssistantResponse(content="Direct response")
        assert resp.content == "Direct response"
        assert resp.type == "assistant"

        tool = ToolCall(tool_name="tool", result="result", success=False)
        assert tool.success is False

        comp = Compaction(summary="Summary", items_compacted=3)
        assert comp.items_compacted == 3


class TestTypeCheckFiltering:
    """Test filtering mixed lists by type with isinstance."""

    def test_filter_by_type(self):
        items: list[MemoryItemType] = [
            UserMessage(content="Hello"),
            AssistantResponse(content="Hi"),
            ToolCall(tool_name="think", result="..."),
            UserMessage(content="How are you?"),
            Compaction(summary="Earlier context"),
        ]

        user_msgs = [item for item in items if isinstance(item, UserMessage)]
        assert len(user_msgs) == 2
        assert user_msgs[0].content == "Hello"
        assert user_msgs[1].content == "How are you?"

        asst_msgs = [item for item in items if isinstance(item, AssistantResponse)]
        assert len(asst_msgs) == 1
        assert asst_msgs[0].content == "Hi"

        tools = [item for item in items if isinstance(item, ToolCall)]
        assert len(tools) == 1
        assert tools[0].tool_name == "think"

        compactions = [item for item in items if isinstance(item, Compaction)]
        assert len(compactions) == 1
        assert compactions[0].summary == "Earlier context"


class TestParse:
    """Test MemoryItem.parse classmethod."""

    def test_parse_user_message(self):
        item = MemoryItem.parse("U: Hello world")
        assert isinstance(item, UserMessage)
        assert item.content == "Hello world"

    def test_parse_assistant_response(self):
        item = MemoryItem.parse("A: Hi there")
        assert isinstance(item, AssistantResponse)
        assert item.content == "Hi there"

    def test_parse_tool_call(self):
        item = MemoryItem.parse("T[read_files]: some result")
        assert isinstance(item, ToolCall)
        assert item.tool_name == "read_files"
        assert item.result == "some result"

    def test_parse_compaction(self):
        item = MemoryItem.parse("C: Summarized content")
        assert isinstance(item, Compaction)
        assert item.summary == "Summarized content"

    def test_parse_empty_returns_none(self):
        assert MemoryItem.parse("") is None
        assert MemoryItem.parse("   ") is None

    def test_parse_unknown_format_returns_none(self):
        assert MemoryItem.parse("X: something") is None

    def test_parse_strips_whitespace(self):
        item = MemoryItem.parse("  U: Hello  ")
        assert isinstance(item, UserMessage)
        assert item.content == "Hello"


class TestTimestamps:
    """Test timestamp handling."""

    def test_timestamp_auto_populated(self):
        msg = UserMessage(content="Test")
        assert msg.timestamp is not None
        assert isinstance(msg.timestamp, datetime)

    def test_timestamp_can_be_overridden(self):
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        msg = UserMessage(content="Test", timestamp=custom_time)
        assert msg.timestamp == custom_time

    def test_all_types_have_timestamps(self):
        user_msg = UserMessage(content="Test")
        asst_msg = AssistantResponse(content="Test")
        tool_msg = ToolCall(tool_name="tool", result="result")
        comp_msg = Compaction(summary="Summary")

        for item in [user_msg, asst_msg, tool_msg, comp_msg]:
            assert hasattr(item, "timestamp")
            assert isinstance(item.timestamp, datetime)


class TestStringRepresentation:
    """Test that string representation maintains backward compatibility."""

    def test_user_message_format(self):
        msg = UserMessage(content="Hello world")
        assert str(msg) == "U: Hello world"

    def test_assistant_response_format(self):
        resp = AssistantResponse(content="Hi there")
        assert str(resp) == "A: Hi there"

    def test_tool_call_format(self):
        tool = ToolCall(tool_name="read_files", result="file contents here")
        assert str(tool) == "T[read_files]: file contents here"

    def test_compaction_format(self):
        comp = Compaction(summary="Summarized context")
        assert str(comp) == "C: Summarized context"

    def test_empty_content(self):
        """Test handling of empty strings."""
        msg = UserMessage(content="")
        assert str(msg) == "U: "

        resp = AssistantResponse(content="")
        assert str(resp) == "A: "

        tool = ToolCall(tool_name="tool", result="")
        assert str(tool) == "T[tool]: "

    def test_multiline_content(self):
        """Test items with multiline content."""
        content = "Line 1\nLine 2\nLine 3"
        msg = UserMessage(content=content)
        assert str(msg) == f"U: {content}"
        assert "Line 1\nLine 2\nLine 3" in str(msg)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_tool_call_with_success_false(self):
        tool = ToolCall(
            tool_name="failing_tool", result="Error occurred", success=False
        )
        assert tool.success is False
        assert str(tool) == "T[failing_tool]: Error occurred"

    def test_items_with_special_characters(self):
        """Test handling of special characters in content."""
        msg = UserMessage(content="Message with: colons and [brackets] and {braces}")
        assert "colons" in msg.content
        assert "[brackets]" in msg.content

    def test_type_field_not_in_init(self):
        """Verify that type field cannot be overridden at initialization."""
        msg = UserMessage(content="test")
        assert msg.type == "user"

    def test_items_compacted_tracking(self):
        """Test that compaction tracks number of items compacted."""
        comp1 = Compaction(summary="Summary", items_compacted=10)
        comp2 = Compaction(summary="Another summary", items_compacted=25)

        assert comp1.items_compacted == 10
        assert comp2.items_compacted == 25
