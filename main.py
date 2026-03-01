import asyncio

from pygents import ContextItem, Turn

from app.agent.tools.think import think
from app.core.factories import get_agent, get_working_memory
from app.memory import UserMessage


async def run_agent():
    agent = await get_agent()
    memory = await get_working_memory()
    agent.context_queue = memory
    while True:
        message = input("\033[90mYou:\033[0m ")
        if message.lower() in ["exit", "quit"]:
            break

        await agent.context_pool.clear()
        await memory.append(ContextItem(UserMessage(content=message)))

        first_turn = Turn(think)
        await agent.put(first_turn)

        async for _, output in agent.run():
            if isinstance(output, str):
                print(output, end="", flush=True)
        print(end="\n")


if __name__ == "__main__":
    asyncio.run(run_agent())
