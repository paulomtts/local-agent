from typing import Any

import tiktoken
from py_ai_toolkit import PyAIToolkit
from py_ai_toolkit.core.domain.interfaces import LLMConfig

from app.core.factories import get_working_memory
from app.core.logger import logger
from app.memory.dataclasses import Compaction, MemoryItemType

COMPACT_PROMPT = """Summarize the following conversation and context into a single consolidated summary. Preserve all information that is relevant for the assistant to continue helping the user (recent intents, file contents or paths mentioned, decisions, tool outcomes). Omit only redundant or purely decorative detail. Output the summary only, no preamble.

# Context
{{ context }}
"""


def token_count(items: list[Any]) -> int:
    text = "\n".join(str(item) for item in items)
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


async def compact_memory(items: list[MemoryItemType | Any]):
    """Compact older memory items into a summary, keeping recent items intact.

    Args:
        items: List of memory items (MemoryItemType objects)
    """
    logger.debug(f"\033[93m[TASK:compact_memory ({token_count(items)} tokens)]\033[0m")

    KEEP_RECENT = 5

    if len(items) <= KEEP_RECENT:
        logger.debug("[TASK:compact_memory] Not enough items to compact, skipping.")
        return

    old_items = items[:-KEEP_RECENT]
    recent_items = items[-KEEP_RECENT:]

    logger.debug(
        f"[TASK:compact_memory] Compacting {len(old_items)} old items, keeping {len(recent_items)} recent"
    )
    config = LLMConfig()
    toolkit = PyAIToolkit(config)
    context = "\n".join(str(item) for item in old_items)
    output = await toolkit.chat(
        template=COMPACT_PROMPT,
        context=context,
    )

    memory = get_working_memory()
    compacted = Compaction(summary=output.content, items_compacted=len(old_items))
    memory.items = [compacted] + recent_items
