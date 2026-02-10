import asyncio
from dotenv import load_dotenv
from py_ai_toolkit import PyAIToolkit
from py_ai_toolkit.core.domain.interfaces import LLMConfig
from pygents import Agent, Turn, tool

from app.factories import get_terminal_session
from app.logic.read_files import read_files_tool_impl
from app.logic.think import think_tool_impl

load_dotenv()

config = LLMConfig()
toolkit = PyAIToolkit(config)
terminal_session = get_terminal_session()


@tool()
async def read_files(context: str):
    "Use to find and read relevant files."
    file_contents = await read_files_tool_impl(
        terminal_session=terminal_session,
        context=context,
        toolkit=toolkit,
    )
    context += "read_files tool used:\n" + file_contents
    return Turn(respond, args=[context])


@tool
async def respond(context: str):
    "Use to respond to the user's message."
    async for chunk in toolkit.stream(
        template=f"Respond to the user's message according to the following context: {context}"
    ):
        yield chunk.content


@tool
async def think(message: str):
    return await think_tool_impl(message=message, toolkit=toolkit)


async def run_agent():
    agent = Agent(
        name="Local Agent",
        description="A helpful assistant with access to tools.",
        tools=[read_files, respond, think],
    )
    while True:
        message = input("You: ")
        if message.lower() in ["exit", "quit"]:
            break
        first_turn = Turn(think, kwargs=dict(message=message))
        await agent.put(first_turn)
        print("Agent: ", end="")
        async for first_turn, output in agent.run():
            if isinstance(output, str):
                print(output, end="", flush=True)
        print()
    terminal_session.close()


if __name__ == "__main__":
    asyncio.run(run_agent())
