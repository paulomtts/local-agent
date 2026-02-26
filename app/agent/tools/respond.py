import sys
from pathlib import Path

from pygents import ContextItem, ContextQueue, tool

from app.agent.utils.definitions import get_tools_definitions
from app.core.factories import get_toolkit
from app.core.logger import log_token_usage, log_tool_use
from app.memory import AssistantResponse, get_recent_episodic_events

SEMANTIC_FILE = Path(__file__).resolve().parents[3] / ".memory" / "semantic.md"

RESPOND_PROMPT = """You are a helpful assistant. Use the background knowledge and episodic memory when relevant, and respond naturally to the user's latest message based on the recent conversation.

# Available Tools
{{ tools }}

# Semantic Memory (Background Facts)
{{ semantic_facts }}

# Episodic Memory (Recent Interactions)
{{ episodic_events }}

# Working Memory (Recent Conversation)
{{ conversation_pairs }}

Respond to the user's latest message."""


async def generate_assistant_response(memory: ContextQueue):
    conversation_pairs = "\n\n".join(str(item.content) for item in memory.items)
    tools = get_tools_definitions()

    semantic_facts = "(none)"
    if SEMANTIC_FILE.exists():
        content = SEMANTIC_FILE.read_text().strip()
        if content:
            semantic_facts = content

    episodic_events = get_recent_episodic_events(n=5)
    if not episodic_events:
        episodic_events = "(none)"

    toolkit = get_toolkit()

    last_chunk = None
    async for chunk in toolkit.stream(
        template=RESPOND_PROMPT,
        conversation_pairs=conversation_pairs,
        tools=tools,
        semantic_facts=semantic_facts,
        episodic_events=episodic_events,
    ):
        last_chunk = chunk
        yield chunk.content

    if last_chunk:
        log_token_usage(
            "respond", last_chunk
        )  # TODO: should improve to allow list of chunks


@tool
async def respond(memory: ContextQueue):
    log_tool_use("respond")
    full_response = ""
    async for chunk in generate_assistant_response(memory=memory):
        if not full_response:
            yield "⤷ "
        full_response += chunk
        yield chunk
    sys.stdout.write("\n")
    await memory.append(ContextItem(AssistantResponse(content=full_response)))
