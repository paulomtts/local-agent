import sys

from pygents import ContextItem, ContextPool, ContextQueue, ToolRegistry, tool

from app.core.factories import get_toolkit
from app.core.logger import log_tool_use
from app.memory import (
    AssistantResponse,
    get_episodic_events,
    get_pool_context,
    get_semantic_facts,
)

RESPOND_PROMPT = """You are a helpful assistant. Use the background knowledge and episodic memory when relevant, and respond naturally to the user's latest message based on the recent conversation.

# Available Tools
{{ tools }}

# Semantic Memory (Background Facts)
{{ semantic_facts }}

# Episodic Memory (Recent Interactions)
{{ episodic_events }}

# Working Memory (Recent Conversation)
{{ working_memory }}

# Fetched Context (Tool Outputs)
{{ pool_context }}

Rules:
- Respond to the user's latest message in humanized prose.
- If provided with JSON data, format it into a humanized prose response."""


@tool
async def respond(
    memory: ContextQueue,
    pool: ContextPool,
    context_ids: list[str] | None = None,
):
    log_tool_use("respond")
    toolkit = get_toolkit()
    full_response = ""
    async for chunk in toolkit.stream(
        template=RESPOND_PROMPT,
        tools=ToolRegistry.definitions(),
        working_memory=memory.history(),
        semantic_facts=get_semantic_facts() or "(none)",
        episodic_events=get_episodic_events(n=5) or "(none)",
        pool_context=get_pool_context(pool, context_ids) if context_ids else "(none)",
    ):
        if not full_response:
            yield "⤷ "
        full_response += chunk.content
        yield chunk.content
    sys.stdout.write("\n")
    # TODO: log token usage
    yield ContextItem(AssistantResponse(content=full_response))
