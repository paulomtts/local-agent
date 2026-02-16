from py_ai_toolkit import PyAIToolkit
from pydantic import BaseModel, Field
from pygents import Memory, Turn, tool

from app.agent.tools.think import think
from app.agent.utils.file_search import (
    deduplicate_paths,
    read_file_contents,
    search_files_by_content,
    search_files_by_name,
)
from app.core.factories import get_toolkit
from app.memory import format_tool_call, get_user_messages_only

RELEVANT_KEYWORDS_PROMPT = """You must generate the relevant keywords to search for based on the context.

# Context
{{ context }}
"""


class GenerateRelevantKeywords(BaseModel):
    "Use this to generate relevant keywords for a file search. Start with smaller words, expanding into longer expressions."

    keywords: list[str] = Field(
        description="The relevant keywords to search for. Each keyword must be snake_case, containing only lowercase letters, numbers, or underscores, and must not contain any whitespace.",
        examples=[
            "word",
            "snake_case",
        ],
        min_length=1,
        max_length=10,
    )


async def get_file_contents(
    memory: Memory,
    toolkit: PyAIToolkit,
) -> str:
    context = get_user_messages_only(memory, n=5)
    search = await toolkit.asend(
        response_model=GenerateRelevantKeywords,
        template=RELEVANT_KEYWORDS_PROMPT,
        context=context,
    )
    keywords = [
        keyword.strip() for keyword in search.content.keywords if keyword.strip()
    ]
    if not keywords:
        return "No relevant keywords found."

    file_names = search_files_by_name(keywords)
    file_names += search_files_by_content(keywords)
    file_names = deduplicate_paths(file_names)

    if not file_names:
        return "No relevant files found."

    result = read_file_contents(file_names)
    return result or "No content read from files."


@tool()
async def read_files(memory: Memory):
    "Use to find and read relevant files."
    toolkit = get_toolkit()
    file_contents = await get_file_contents(
        memory=memory,
        toolkit=toolkit,
    )
    await memory.append(format_tool_call("read_files", file_contents))
    return Turn(think, args=[memory])
