import tiktoken
from grafo import Node, TreeExecutor
from pygents import ContextItem

from app.core.config import WORKING_MEMORY_TOKEN_THRESHOLD
from app.core.logger import log_hook
from app.memory.compact import compact_memory
from app.memory.dataclasses import MemoryItem, UserMessage
from app.memory.extractors import (
    extract_episodic_memory,
    extract_semantic_memory,
)
from app.memory.queries import WORKING_FILE


def _token_count(items: list[ContextItem]) -> int:
    text = "\n".join(str(item.content) for item in items)
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _latest_user_message(items: list[ContextItem[MemoryItem]]) -> str | None:
    """Return the most recent user message content, or None if not found."""
    for item in reversed(items):
        content = item.content
        if isinstance(content, UserMessage):
            return content.content
    return None


def _is_trivial_message(msg: str) -> bool:
    trivial_patterns = {
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
    }
    return msg.lower().strip() in trivial_patterns or len(msg.split()) <= 2


def write_working_memory(items: list[ContextItem[MemoryItem]], added: int) -> None:
    content = "\n\n---\n\n".join(str(item.content) for item in items)
    with WORKING_FILE.open("w") as f:
        f.write(content)
    log_hook("after_append", "write", f"working +{added}")


def _appended_contains_user_message(appended_items: list[ContextItem]) -> bool:
    for item in appended_items:
        if isinstance(getattr(item, "content", None), UserMessage):
            return True
    return False


def build_tree(
    items: list[ContextItem[MemoryItem]],
    appended_items: list[ContextItem] | None = None,
    queue=None,
) -> TreeExecutor | None:
    msg = _latest_user_message(items)
    if msg is None:
        return None
    message_is_trivial = _is_trivial_message(msg)
    above_threshold = _token_count(items) >= WORKING_MEMORY_TOKEN_THRESHOLD
    run_extraction = (
        appended_items is not None and _appended_contains_user_message(appended_items)
    ) or (appended_items is None)

    if message_is_trivial and not above_threshold and not run_extraction:
        return None

    roots: list[Node] = []

    if above_threshold and queue is not None:
        roots.append(
            Node[None](
                coroutine=compact_memory,
                uuid="compact_memory",
                kwargs={"memory": queue, "items": items},
            )
        )

    if run_extraction and not message_is_trivial:
        roots.append(
            Node[None](
                coroutine=extract_semantic_memory,
                uuid="semantic_memory",
                kwargs={"items": items, "user_message": msg},
            )
        )
        roots.append(
            Node[None](
                coroutine=extract_episodic_memory,
                uuid="episodic_memory",
                kwargs={"user_message": msg},
            )
        )

    return TreeExecutor(roots=roots) if roots else None
