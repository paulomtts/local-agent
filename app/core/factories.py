from functools import lru_cache

from py_ai_toolkit import PyAIToolkit
from pygents import Agent, Memory


@lru_cache
def get_toolkit():
    return PyAIToolkit()


@lru_cache
def get_working_memory():
    """Get the working memory, which includes a hook to compact the memory if it exceeds the token threshold."""
    from app.memory import after_append

    return Memory(limit=40, hooks=[after_append])


@lru_cache
def get_agent():
    from app.agent.tools.read_files import read_files
    from app.agent.tools.respond import respond
    from app.agent.tools.think import think

    return Agent(
        name="Local Agent",
        description="A helpful assistant with access to tools.",
        tools=[read_files, respond, think],
    )
