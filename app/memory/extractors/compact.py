from typing import Any

import tiktoken
from py_ai_toolkit import PyAIToolkit
from py_ai_toolkit.core.domain.interfaces import LLMConfig

from app.core.factories import get_working_memory
from app.core.logger import logger
from app.memory.format import format_compaction

COMPACT_PROMPT = """Summarize the following conversation and context into a single consolidated summary. Preserve all information that is relevant for the assistant to continue helping the user (recent intents, file contents or paths mentioned, decisions, tool outcomes). Omit only redundant or purely decorative detail. Output the summary only, no preamble.

# Context
{{ context }}
"""


def token_count(items: list[Any]) -> int:
    text = "\n".join(str(item) for item in items)
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


async def compact_memory(items: list[Any]):
    logger.debug(f"\033[93m[TASK:compact_memory ({token_count(items)} tokens)]\033[0m")

    # Keep last 5 items, compact the rest
    KEEP_RECENT = 5

    if len(items) <= KEEP_RECENT:
        logger.debug("[TASK:compact_memory] Not enough items to compact, skipping.")
        return

    old_items = items[:-KEEP_RECENT]
    recent_items = items[-KEEP_RECENT:]

    # Compact only old items
    logger.debug(f"[TASK:compact_memory] Compacting {len(old_items)} old items, keeping {len(recent_items)} recent")
    config = LLMConfig()
    toolkit = PyAIToolkit(config)
    context = "\n".join(str(item) for item in old_items)
    output = await toolkit.chat(
        template=COMPACT_PROMPT,
        context=context,
    )

    # Replace memory with: [compacted_summary] + recent_items
    memory = get_working_memory()
    compacted = format_compaction(output.content)
    memory.items = [compacted] + recent_items
