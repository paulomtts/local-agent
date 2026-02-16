from pathlib import Path

from py_ai_toolkit import PyAIToolkit
from pygents import Memory, tool

from app.core.factories import get_toolkit
from app.core.logger import log_prompt
from app.memory import AssistantResponse, get_conversation_pairs, get_recent_episodic_events

SEMANTIC_FILE = Path(__file__).resolve().parents[3] / ".memory" / "semantic.md"

RESPOND_PROMPT = """You are a helpful assistant. Use the background knowledge and episodic memory when relevant, and respond naturally to the user's latest message based on the recent conversation.

# Semantic Memory (Background Facts)
{{ semantic_facts }}

# Episodic Memory (Recent Interactions)
{{ episodic_events }}

# Working Memory (Recent Conversation)
{{ conversation_pairs }}

Respond to the user's latest message:"""


async def generate_assistant_response(memory: Memory, toolkit: PyAIToolkit):
    conversation_pairs = get_conversation_pairs(memory, n=3)

    semantic_facts = "(none)"
    if SEMANTIC_FILE.exists():
        content = SEMANTIC_FILE.read_text().strip()
        if content:
            semantic_facts = content

    episodic_events = get_recent_episodic_events(n=5)
    if not episodic_events:
        episodic_events = "(none)"

    last_chunk = None
    async for chunk in toolkit.stream(
        template=RESPOND_PROMPT,
        conversation_pairs=conversation_pairs,
        semantic_facts=semantic_facts,
        episodic_events=episodic_events,
    ):
        last_chunk = chunk
        yield chunk.content

    # Log token usage from final chunk
    if last_chunk:
        log_prompt("respond", last_chunk)


@tool
async def respond(memory: Memory):
    "Use to respond to the user's message."
    toolkit = get_toolkit()
    full_response = ""
    async for chunk in generate_assistant_response(memory=memory, toolkit=toolkit):
        if not full_response:
            yield "⤷ "
        full_response += chunk
        yield chunk
    await memory.append(AssistantResponse(content=full_response))
