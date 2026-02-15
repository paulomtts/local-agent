import asyncio

from pygents import Turn

from app.factories import get_agent, get_working_memory
from app.tools.think import think


async def run_agent():
    agent = get_agent()
    memory = get_working_memory()
    while True:
        message = input("You: ")
        if message.lower() in ["exit", "quit"]:
            break
        await memory.append(f"User message: {message}")
        first_turn = Turn(think, args=[memory])
        await agent.put(first_turn)
        print("Agent: ", end="")
        async for _, output in agent.run():
            if isinstance(output, str):
                print(output, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(run_agent())
