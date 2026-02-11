import asyncio
from dotenv import load_dotenv
from py_ai_toolkit import PyAIToolkit
from py_ai_toolkit.core.domain.interfaces import LLMConfig
from pygents import Agent, Turn, Memory, tool, ToolRegistry

from app.hooks.compact import compact_memory
from app.tools.read_files import get_file_contents
from app.tools.think import decide_next_tool

load_dotenv()

config = LLMConfig()
toolkit = PyAIToolkit(config)


@tool()
async def read_files(memory: Memory):
    "Use to find and read relevant files."
    context = "\n".join(str(item) for item in memory)
    file_contents = await get_file_contents(
        context=context,
        toolkit=toolkit,
    )
    await memory.append(f"read_files tool used:\n{file_contents}")
    return Turn(think, args=[memory])


@tool
async def respond(memory: Memory):
    "Use to respond to the user's message."
    context = "\n".join(str(item) for item in memory)
    async for chunk in toolkit.stream(
        template=f"Respond to the user's message according to the following context: {context}"
    ):
        yield chunk.content


@tool
async def think(memory: Memory):
    tool_name = await decide_next_tool(memory=memory, toolkit=toolkit)
    target_tool = ToolRegistry.get(tool_name)
    return Turn(target_tool, args=[memory])


async def run_agent():
    agent = Agent(
        name="Local Agent",
        description="A helpful assistant with access to tools.",
        tools=[read_files, respond, think],
    )
    memory = Memory(limit=20, hooks=[compact_memory])
    while True:
        message = input("You: ")
        if message.lower() in ["exit", "quit"]:
            break
        await memory.append(f"User message: {message}")
        first_turn = Turn(think, args=[memory])
        await agent.put(first_turn)
        print("Agent: ", end="")
        async for first_turn, output in agent.run():
            if isinstance(output, str):
                print(output, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(run_agent())
