from typing import Any

import tiktoken
from py_ai_toolkit import PyAIToolkit
from py_ai_toolkit.core.domain.interfaces import LLMConfig

from app.factories import get_working_memory
from app.logger import logger

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
    config = LLMConfig()
    toolkit = PyAIToolkit(config)
    context = "\n".join(str(item) for item in items)
    output = await toolkit.chat(
        template=COMPACT_PROMPT,
        context=context,
    )
    memory = get_working_memory()
    memory.items = [output.content]
