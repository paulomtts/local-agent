from typing import Any

import tiktoken
from py_ai_toolkit import PyAIToolkit
from py_ai_toolkit.core.domain.interfaces import LLMConfig

from app.core.factories import get_working_memory
from app.core.logger import RESET, TASK_TAG, logger, log_prompt
from app.memory.dataclasses import Compaction, MemoryItemType

COMPACT_PROMPT = """Summarize the following conversation items into a single consolidated summary. Preserve all information that is relevant for the assistant to continue helping the user (recent intents, file contents or paths mentioned, decisions, tool outcomes). Omit only redundant or purely decorative detail. Output the summary only, no preamble.

# Working Memory (Items to Compact)
{{ old_items }}
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
    KEEP_RECENT = 5

    if len(items) <= KEEP_RECENT:
        logger.debug(f"{TASK_TAG}[TASK:compact_memory]{RESET} skip")
        return

    old_items = items[:-KEEP_RECENT]
    recent_items = items[-KEEP_RECENT:]

    logger.debug(f"{TASK_TAG}[TASK:compact_memory]{RESET} {len(old_items)}→{len(recent_items)} items")

    config = LLMConfig()
    toolkit = PyAIToolkit(config)
    old_items_text = "\n".join(str(item) for item in old_items)
    output = await toolkit.chat(
        template=COMPACT_PROMPT,
        old_items=old_items_text,
    )

    log_prompt("compact", output)

    memory = get_working_memory()
    compacted = Compaction(summary=output.content, items_compacted=len(old_items))
    memory.items = [compacted] + recent_items
