from functools import lru_cache

from py_ai_toolkit import PyAIToolkit
from pygents import Agent, ContextItem, ContextQueue

from app.core.logger import log_hook


@lru_cache
def get_toolkit():
    return PyAIToolkit()


@lru_cache
async def get_working_memory() -> ContextQueue:
    """Get the working memory.

    Returns:
        ContextQueue: Working memory limited to 40 items. Items stored are MemoryItemType
            objects (UserMessage, AssistantResponse, ToolCall, or Compaction).
            Includes a hook to compact memory if it exceeds the token threshold.
            On first initialization, loads previous conversation from working.md.
    """
    from app.memory import get_working_memory
    from app.memory.hooks import LOADING_WORKING_MEMORY

    memory = ContextQueue(limit=40)
    items = get_working_memory()
    if items:
        token = LOADING_WORKING_MEMORY.set(True)
        try:
            await memory.append(*[ContextItem(item) for item in items])
            log_hook("working", "load", f"{len(items)} items")
        finally:
            LOADING_WORKING_MEMORY.reset(token)
    else:
        log_hook("working", "load", "fresh")

    return memory


@lru_cache
async def get_agent():
    from app.agent.tools import calendar, orchestrate, read_files, respond, think

    return Agent(
        name="Local Agent",
        description="A helpful assistant with access to tools.",
        tools=[calendar, orchestrate, read_files, respond, think],
    )
