from functools import lru_cache

from py_ai_toolkit import PyAIToolkit
from pygents import Agent, Memory

from logging import INFO, getLogger, StreamHandler

logger = getLogger("local_agent")
logger.setLevel(INFO)
logger.addHandler(StreamHandler())


@lru_cache
def get_toolkit():
    return PyAIToolkit()


@lru_cache
def get_working_memory():
    """Get the working memory, which includes a hook to compact the memory if it exceeds the token threshold."""
    from app.hooks.memory import compact_memory

    return Memory(limit=20, hooks=[compact_memory])


@lru_cache
def get_agent():
    from app.tools.read_files import read_files
    from app.tools.respond import respond
    from app.tools.think import think

    return Agent(
        name="Local Agent",
        description="A helpful assistant with access to tools.",
        tools=[read_files, respond, think],
    )
