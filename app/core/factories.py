from functools import lru_cache

from py_ai_toolkit import PyAIToolkit
from pygents import Agent, Memory


@lru_cache
def get_toolkit():
    return PyAIToolkit()


@lru_cache
def get_working_memory() -> Memory:
    """Get the working memory.

    Returns:
        Memory: Working memory limited to 40 items. Items stored are MemoryItemType
            objects (UserMessage, AssistantResponse, ToolCall, or Compaction).
            Includes a hook to compact memory if it exceeds the token threshold.
            On first initialization, loads previous conversation from working.md.
    """
    from app.memory import after_append, load_working_memory

    memory = Memory(limit=40, hooks=[after_append])
    items = load_working_memory()
    if items:
        memory.items.extend(items)

    return memory


@lru_cache
def get_agent():
    from app.agent.tools import read_files, respond, think

    return Agent(
        name="Local Agent",
        description="A helpful assistant with access to tools.",
        tools=[read_files, respond, think],
    )
