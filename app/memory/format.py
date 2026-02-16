"""Centralized memory format constants and parsing utilities."""

# Format prefixes
USER_PREFIX = "U:"
ASST_PREFIX = "A:"
TOOL_PREFIX_TEMPLATE = "T[{}]:"
COMPACT_PREFIX = "C:"

# Timestamp formats
EPISODIC_TIMESTAMP_FORMAT = "%m-%d %H:%M"


def format_user_message(message: str) -> str:
    """Format user message with compact notation."""
    return f"{USER_PREFIX} {message}"


def format_assistant_response(response: str) -> str:
    """Format assistant response with compact notation."""
    return f"{ASST_PREFIX} {response}"


def format_tool_call(tool_name: str, result: str) -> str:
    """Format tool call with compact notation."""
    return f"T[{tool_name}]: {result}"


def format_compaction(summary: str) -> str:
    """Format compaction marker."""
    return f"{COMPACT_PREFIX} {summary}"


def extract_message_content(item: str) -> tuple[str, str]:
    """Extract type and content from formatted message.

    Returns:
        (type, content) where type is 'user', 'assistant', 'tool', or 'compact'
    """
    item = str(item)
    if item.startswith(USER_PREFIX):
        return ("user", item.removeprefix(USER_PREFIX).strip())
    elif item.startswith(ASST_PREFIX):
        return ("assistant", item.removeprefix(ASST_PREFIX).strip())
    elif item.startswith("T["):
        end_bracket = item.find("]:")
        if end_bracket != -1:
            content = item[end_bracket + 2 :].strip()
            return ("tool", content)
        return ("tool", item)
    elif item.startswith(COMPACT_PREFIX):
        return ("compact", item.removeprefix(COMPACT_PREFIX).strip())
    else:
        return ("unknown", item)


def is_user_message(item: str) -> bool:
    """Check if item is a user message."""
    msg_type, _ = extract_message_content(item)
    return msg_type == "user"


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
        item_type, _ = extract_message_content(memory.items[i])
        if item_type in ("user", "assistant"):
            pairs.insert(0, memory.items[i])
        i -= 1
    return "\n".join(str(item) for item in pairs)
