from pathlib import Path

from py_ai_toolkit import PyAIToolkit
from pygents import Memory, tool

from app.core.factories import get_toolkit
from app.memory import AssistantResponse, get_conversation_pairs

SEMANTIC_FILE = Path(__file__).resolve().parents[3] / ".memory" / "semantic.md"

RESPOND_PROMPT = """You are a helpful assistant. Use the background knowledge when relevant, and respond naturally to the user's latest message based on the recent conversation.

# Background Knowledge
{{ semantic_facts }}

# Recent Conversation
{{ context }}

Respond to the user's latest message:"""


async def generate_assistant_response(memory: Memory, toolkit: PyAIToolkit):
    context = get_conversation_pairs(memory, n=3)

    semantic_facts = "(none)"
    if SEMANTIC_FILE.exists():
        content = SEMANTIC_FILE.read_text().strip()
        if content:
            semantic_facts = content

    async for chunk in toolkit.stream(
        template=RESPOND_PROMPT,
        context=context,
        semantic_facts=semantic_facts,
    ):
        yield chunk.content


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
