from functools import lru_cache

from py_ai_toolkit import PyAIToolkit
from pygents import Agent, ContextItem, ContextQueue


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

    memory = ContextQueue(limit=40)
    items = get_working_memory()
    if items:
        await memory.append(*[ContextItem(item) for item in items])

    return memory


@lru_cache
async def get_agent():
    from app.agent.tools import read_files, respond, think

    return Agent(
        name="Local Agent",
        description="A helpful assistant with access to tools.",
        tools=[read_files, respond, think],
    )
