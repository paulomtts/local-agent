import asyncio

from pygents import Turn

from app.factories import get_agent, get_working_memory
from app.tools.think import think


async def run_agent():
    agent = get_agent()
    memory = get_working_memory()
    while True:
        message = input("\033[90mYou:\033[0m ")
        if message.lower() in ["exit", "quit"]:
            break
        await memory.append(f"User message: {message}")
        first_turn = Turn(think, args=[memory])
        await agent.put(first_turn)
        # print("\033[96mAgent:\033[0m ", end="")  # Cyan color
        async for _, output in agent.run():
            if isinstance(output, str):
                print(output, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(run_agent())
