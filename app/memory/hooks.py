import sys
from pathlib import Path
from typing import Any

from grafo import Node, TreeExecutor
from pygents import MemoryHook, hook

from app.core.config import WORKING_MEMORY_TOKEN_THRESHOLD
from app.core.logger import HOOK_TAG, RESET, logger
from app.memory.dataclasses import (
    AssistantResponse,
    Compaction,
    MemoryItemType,
    ToolCall,
    UserMessage,
    is_user_message,
)
from app.memory.extractors import (
    compact_memory,
    extract_episodic_memory,
    extract_semantic_memory,
    token_count,
)

WORKING_FILE = Path(__file__).resolve().parents[2] / ".memory" / "working.md"


def _get_current_memory_items() -> list[Any]:
    from app.core.factories import get_working_memory

    return get_working_memory().items


def _extract_user_message(items: list[Any]) -> str:
    if not items:
        raise ValueError("Items list is empty")

    last_item = items[-1]
    if is_user_message(last_item):
        return last_item.content
    raise ValueError("Last item is not a user message")


def _is_trivial_message(msg: str) -> bool:
    """Check if a message is trivial (short acknowledgment)."""
    trivial_patterns = [
        "thanks",
        "thank you",
        "ok",
        "okay",
        "yes",
        "no",
        "sure",
        "got it",
        "understood",
        "bye",
        "hello",
        "hi",
    ]
    msg_lower = msg.lower().strip()
    return msg_lower in trivial_patterns or len(msg.split()) <= 2


def _should_extract_semantic(items: list[Any]) -> bool:
    """Check if semantic memory should be extracted (on user messages)."""
    if not items or not is_user_message(items[-1]):
        return False

    msg = _extract_user_message(items)
    return not _is_trivial_message(msg)


def _should_extract_episodic(items: list[Any]) -> bool:
    """Check if episodic memory should be extracted (on user messages).

    Episodic extraction now happens on user messages to capture user intent.
    Agent actions are logged deterministically by tools themselves.
    """
    if not items or not is_user_message(items[-1]):
        return False

    msg = _extract_user_message(items)
    return not _is_trivial_message(msg)


def _build_tree(items: list[Any]) -> TreeExecutor | None:
    should_extract_sem = _should_extract_semantic(items)
    should_extract_epi = _should_extract_episodic(items)
    above_threshold = token_count(items) >= WORKING_MEMORY_TOKEN_THRESHOLD

    if not should_extract_sem and not should_extract_epi and not above_threshold:
        return None

    roots: list[Node] = []

    if above_threshold:
        roots.append(
            Node[None](
                coroutine=compact_memory, uuid="compact_memory", kwargs={"items": items}
            )
        )

    if should_extract_sem:
        msg = _extract_user_message(items)
        roots.append(
            Node[None](
                coroutine=extract_semantic_memory,
                uuid="semantic_memory",
                kwargs={"items": items, "user_message": msg},
            )
        )

    if should_extract_epi:
        msg = _extract_user_message(items)
        roots.append(
            Node[None](
                coroutine=extract_episodic_memory,
                uuid="episodic_memory",
                kwargs={"user_message": msg},
            )
        )

    return TreeExecutor(roots=roots)


def _parse_working_memory_item(section: str) -> MemoryItemType | None:
    """Parse a working memory section into a typed memory item.

    Args:
        section: A section from working.md (e.g., "U: hello" or "A: response")

    Returns:
        Parsed memory item or None if format is invalid
    """
    section = section.strip()
    if not section:
        return None

    # User message: "U: content"
    if section.startswith("U: "):
        content = section[3:].strip()
        return UserMessage(content=content)

    # Assistant response: "A: content"
    if section.startswith("A: "):
        content = section[3:].strip()
        return AssistantResponse(content=content)

    # Tool call: "T[tool_name]: result"
    if section.startswith("T["):
        try:
            end_bracket = section.index("]:")
            tool_name = section[2:end_bracket]
            result = section[end_bracket + 2 :].strip()
            return ToolCall(tool_name=tool_name, result=result)
        except ValueError:
            return None

    # Compaction: "C: summary"
    if section.startswith("C: "):
        summary = section[3:].strip()
        return Compaction(summary=summary, items_compacted=0)

    return None


def load_working_memory() -> list[MemoryItemType]:
    """Load and parse working memory from working.md file.

    Returns:
        List of parsed memory items, or empty list if file doesn't exist
    """
    if not WORKING_FILE.exists():
        logger.debug(
            f"{HOOK_TAG}[HOOK:load_working]{RESET} file not found, starting fresh"
        )
        return []

    content = WORKING_FILE.read_text().strip()
    if not content:
        logger.debug(f"{HOOK_TAG}[HOOK:load_working]{RESET} empty file, starting fresh")
        return []

    # Split by separator "---"
    sections = content.split("\n\n---\n\n")
    items = []

    for section in sections:
        item = _parse_working_memory_item(section)
        if item:
            items.append(item)

    logger.debug(f"{HOOK_TAG}[HOOK:load_working]{RESET} loaded {len(items)} items")
    return items


def _write_working_memory(items: list[Any], added: int):
    content = "\n\n---\n\n".join(str(item) for item in items)
    with WORKING_FILE.open("w") as f:
        f.write(content)
    logger.debug(f"{HOOK_TAG}[HOOK:after_append:working]{RESET} +{added} item")


@hook(MemoryHook.AFTER_APPEND)
async def after_append(items: list[Any]):
    sys.stdout.write("\n")
    sys.stdout.flush()
    tree = _build_tree(items)
    if tree is None:
        logger.debug(f"{HOOK_TAG}[HOOK:after_append:skip]{RESET}")
        _write_working_memory(items, added=1)
        return items

    try:
        await tree.run()
    except Exception as e:
        logger.error(f"{HOOK_TAG}[HOOK:after_append:error]{RESET} {e}")

    current_items = _get_current_memory_items()
    _write_working_memory(current_items, added=1)
    return current_items
