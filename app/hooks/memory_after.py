from typing import Any

from grafo import Node, TreeExecutor
from pygents import MemoryHook, hook

from app.config import WORKING_MEMORY_TOKEN_THRESHOLD
from app.hooks.tasks.compact import compact_memory, token_count
from app.hooks.tasks.episodic import extract_episodic_memory
from app.hooks.tasks.semantic import extract_semantic_memory
from app.logger import logger


def _is_user_message(items: list[Any]) -> bool:
    if not items:
        return False
    return str(items[-1]).startswith("User message:")


def _extract_user_message(items: list[Any]) -> str:
    return str(items[-1]).removeprefix("User message:").strip()


def _build_tree(items: list[Any]) -> TreeExecutor | None:
    has_user_message = _is_user_message(items)
    above_threshold = token_count(items) >= WORKING_MEMORY_TOKEN_THRESHOLD

    if not has_user_message and not above_threshold:
        return None

    roots: list[Node] = []

    if above_threshold:
        roots.append(
            Node[None](
                coroutine=compact_memory, uuid="compact_memory", kwargs={"items": items}
            )
        )

    if has_user_message:
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


@hook(MemoryHook.AFTER_APPEND)
async def after_append(items: list[Any]):
    tree = _build_tree(items)
    if tree is None:
        logger.debug("[HOOK:after_append] No tasks to run, skipping.")
        return items

    logger.debug(f"[HOOK:after_append] Running {len(tree.roots)} task(s)...")
    try:
        await tree.run()
    except Exception as e:
        logger.error(f"[HOOK:after_append] Error during task execution: {e}")
    return items
