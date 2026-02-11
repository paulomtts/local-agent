import asyncio
import os
import threading
from collections.abc import Callable
from typing import Any

import tiktoken
from pydantic import BaseModel, Field
from py_ai_toolkit import PyAIToolkit
from py_ai_toolkit.core.domain.interfaces import LLMConfig


WORKING_MEMORY_TOKEN_THRESHOLD = int(
    os.environ.get("WORKING_MEMORY_TOKEN_THRESHOLD", "10000")
)

COMPACT_PROMPT = """Summarize the following conversation and context into a single consolidated summary. Preserve all information that is relevant for the assistant to continue helping the user (recent intents, file contents or paths mentioned, decisions, tool outcomes). Omit only redundant or purely decorative detail. Output the summary only, no preamble.

---

{context}

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


def _run_compaction_in_thread(items: list[Any], config: LLMConfig) -> list[str]:
    toolkit = PyAIToolkit(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        summary = loop.run_until_complete(_compact_async(items, toolkit))
        return [summary]
    finally:
        loop.close()


def make_compact_callback(
    config: LLMConfig,
    token_threshold: int = WORKING_MEMORY_TOKEN_THRESHOLD,
) -> Callable[[list[Any]], list[Any]]:
    def compact(items: list[Any]) -> list[Any]:
        if not items:
            return items
        text = "\n".join(str(item) for item in items)
        if _token_count(text) < token_threshold:
            return items
        result: list[Any] = []
        done = threading.Event()

        def in_thread() -> None:
            nonlocal result
            result = _run_compaction_in_thread(items, config)
            done.set()

        thread = threading.Thread(target=in_thread)
        thread.start()
        done.wait()
        return result

    return compact
