from pathlib import Path
from typing import Any

from grafo import Node, TreeExecutor
from pygents import MemoryHook, hook

from app.core.config import WORKING_MEMORY_TOKEN_THRESHOLD
from app.core.logger import logger
from app.memory.extractors.compact import compact_memory, token_count
from app.memory.extractors.episodic import extract_episodic_memory
from app.memory.extractors.semantic import extract_semantic_memory
from app.memory.format import extract_message_content, is_user_message

WORKING_FILE = Path(__file__).resolve().parents[2] / ".memory" / "working.md"


def _get_current_memory_items() -> list[Any]:
    """Get the current items from working memory."""
    from app.core.factories import get_working_memory

    return get_working_memory().items


def _is_user_message(items: list[Any]) -> bool:
    if not items:
        return False
    return is_user_message(items[-1])


def _extract_user_message(items: list[Any]) -> str:
    _, content = extract_message_content(items[-1])
    return content


def _should_extract_memories(items: list[Any]) -> bool:
    """Check if the latest user message warrants memory extraction."""
    if not _is_user_message(items):
        return False

    msg = _extract_user_message(items)

    # Skip trivial messages
    trivial_patterns = [
        "thanks", "thank you", "ok", "okay", "yes", "no",
        "sure", "got it", "understood", "bye", "hello", "hi"
    ]
    msg_lower = msg.lower().strip()

    # Exact match or very short
    if msg_lower in trivial_patterns or len(msg.split()) <= 2:
        return False

    return True


def _build_tree(items: list[Any]) -> TreeExecutor | None:
    has_user_message = _is_user_message(items)
    should_extract = _should_extract_memories(items)
    above_threshold = token_count(items) >= WORKING_MEMORY_TOKEN_THRESHOLD

    if not should_extract and not above_threshold:
        return None

    roots: list[Node] = []

    if above_threshold:
        roots.append(
            Node[None](
                coroutine=compact_memory, uuid="compact_memory", kwargs={"items": items}
            )
        )

    if should_extract:
        msg = _extract_user_message(items)
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
                kwargs={"items": items, "user_message": msg},
            )
        )

    return TreeExecutor(roots=roots)


def _write_working_memory(items: list[Any]):
    """Write the current working memory items to the working.md file."""
    content = "\n\n---\n\n".join(str(item) for item in items)
    with WORKING_FILE.open("w") as f:
        f.write(content)
    logger.debug(f"[HOOK:after_append] Written {len(items)} items to {WORKING_FILE}")


@hook(MemoryHook.AFTER_APPEND)
async def after_append(items: list[Any]):
    tree = _build_tree(items)
    if tree is None:
        logger.debug("[HOOK:after_append] No tasks to run, skipping.")
        _write_working_memory(items)
        return items

    logger.debug(f"[HOOK:after_append] Running {len(tree.roots)} task(s)...")
    try:
        await tree.run()
    except Exception as e:
        logger.error(f"[HOOK:after_append] Error during task execution: {e}")

    # Get the current memory state (in case compaction happened)
    current_items = _get_current_memory_items()
    _write_working_memory(current_items)
    return current_items
