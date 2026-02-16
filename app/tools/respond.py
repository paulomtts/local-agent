from py_ai_toolkit import PyAIToolkit
from pygents import Memory, tool

from app.factories import get_toolkit


async def generate_assistant_response(memory: Memory, toolkit: PyAIToolkit):
    context = "\n".join(str(item) for item in memory)
    async for chunk in toolkit.stream(
        template=f"Respond to the user's message according to the following context: {context}"
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
    await memory.append(f"Assistant response: {full_response}")
