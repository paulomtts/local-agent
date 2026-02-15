from typing import Any

import tiktoken
from py_ai_toolkit import PyAIToolkit
from py_ai_toolkit.core.domain.interfaces import LLMConfig
from pygents import MemoryHook, hook

from app.config import WORKING_MEMORY_TOKEN_THRESHOLD
from app.factories import get_working_memory, logger

COMPACT_PROMPT = """Summarize the following conversation and context into a single consolidated summary. Preserve all information that is relevant for the assistant to continue helping the user (recent intents, file contents or paths mentioned, decisions, tool outcomes). Omit only redundant or purely decorative detail. Output the summary only, no preamble.

# Context
{{ context }}
"""


def _token_count(items: list[Any]) -> int:
    text = "\n".join(str(item) for item in items)
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


async def _compact_memory(items: list[Any]) -> str:
    config = LLMConfig()
    toolkit = PyAIToolkit(config)
    context = "\n".join(str(item) for item in items)
    output = await toolkit.chat(
        template=COMPACT_PROMPT,
        context=context,
    )
    return output.content


async def _store_in_memory(summary: str):
    memory = get_working_memory()
    memory.items = [summary]


@hook(MemoryHook.AFTER_APPEND)
async def compact_memory(items: list[Any]):
    """Compact the memory if it exceeds the token threshold."""
    if _token_count(items) < WORKING_MEMORY_TOKEN_THRESHOLD:
        logger.debug(
            f"Memory count ({_token_count(items)}) is below the token threshold, skipping compacting."
        )
        return items

    logger.debug(f"Compacting memory ({_token_count(items)} items)...")
    summary = await _compact_memory(items)
    await _store_in_memory(summary)
    logger.debug(f"Memory compacted. Result: {summary}")
