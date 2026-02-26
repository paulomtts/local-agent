from py_ai_toolkit import PyAIToolkit
from pydantic import BaseModel, Field
from pygents import ContextQueue, Turn, tool

from app.agent.tools.think import think
from app.agent.utils.file_search import (
    deduplicate_paths,
    read_file_contents,
    search_files_by_content,
    search_files_by_name,
)
from app.core.factories import get_toolkit
from app.core.logger import log_prompt
from app.memory import get_user_messages_only, log_episodic_event

RELEVANT_KEYWORDS_PROMPT = """You must generate the relevant keywords to search for based on the user's messages.

# Working Memory (User Messages)
{{ user_messages }}
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
    memory: ContextQueue,
    toolkit: PyAIToolkit,
) -> str:
    user_messages = get_user_messages_only(memory, n=5)

    search = await toolkit.asend(
        response_model=GenerateRelevantKeywords,
        template=RELEVANT_KEYWORDS_PROMPT,
        user_messages=user_messages,
    )

    log_prompt("read_files", search)
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
async def read_files(memory: ContextQueue):
    "Use to find and read relevant files."
    toolkit = get_toolkit()
    file_contents = await get_file_contents(
        memory=memory,
        toolkit=toolkit,
    )

    # Deterministically log file reading event
    if file_contents and "No relevant files found" not in file_contents:
        # Extract file names from the result for context
        lines = file_contents.split("\n")
        file_mentions = [line for line in lines if line.startswith("# File:")]
        file_count = len(file_mentions)

        if file_count > 0:
            # Get up to 3 file names for context
            file_names = []
            for mention in file_mentions[:3]:
                # Extract filename from "# File: /path/to/file.py"
                if ":" in mention:
                    path = mention.split(":", 1)[1].strip()
                    file_names.append(path.split("/")[-1])

            context = ", ".join(file_names) if file_names else None
            log_episodic_event(
                event=f"agent read {file_count} file{'s' if file_count > 1 else ''}",
                context=context,
            )

    return Turn(think, args=[memory], kwargs={"tool_context": file_contents})
