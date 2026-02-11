import asyncio
import os
import threading
from typing import Any

import tiktoken
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from py_ai_toolkit import PyAIToolkit
from py_ai_toolkit.core.domain.interfaces import LLMConfig
from pygents import hook, MemoryHook

load_dotenv()
_config = LLMConfig()

WORKING_MEMORY_TOKEN_THRESHOLD = int(
    os.environ.get("WORKING_MEMORY_TOKEN_THRESHOLD", "10000")
)

COMPACT_PROMPT = """Summarize the following conversation and context into a single consolidated summary. Preserve all information that is relevant for the assistant to continue helping the user (recent intents, file contents or paths mentioned, decisions, tool outcomes). Omit only redundant or purely decorative detail. Output the summary only, no preamble.

---

{{ context }}

---

Summary:"""


class CompactResult(BaseModel):
    summary: str = Field(
        description="Consolidated summary of the conversation and context."
    )


def _token_count(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


async def _compact_async(items: list[Any], toolkit: PyAIToolkit) -> str:
    context = "\n".join(str(item) for item in items)
    output = await toolkit.asend(
        response_model=CompactResult,
        template=COMPACT_PROMPT,
        context=context,
    )
    return output.content.summary


def _run_compaction_in_thread(items: list[Any]) -> list[str]:
    toolkit = PyAIToolkit(_config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        summary = loop.run_until_complete(_compact_async(items, toolkit))
        return [summary]
    finally:
        loop.close()


@hook(
    MemoryHook.BEFORE_APPEND,
    token_threshold=WORKING_MEMORY_TOKEN_THRESHOLD,
)
async def compact_memory(
    items: list[Any],
    *,
    token_threshold: int,
) -> None:
    if not items:
        return
    text = "\n".join(str(item) for item in items)
    if _token_count(text) < token_threshold:
        return
    done = threading.Event()
    compacted: list[Any] = []

    def in_thread() -> None:
        nonlocal compacted
        compacted = _run_compaction_in_thread(items)
        done.set()

    thread = threading.Thread(target=in_thread)
    thread.start()
    done.wait()
