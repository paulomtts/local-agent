from pydantic import BaseModel, Field
from py_ai_toolkit import PyAIToolkit

from app.utils import (
    deduplicate_paths,
    read_file_contents,
    search_files_by_content,
    search_files_by_name,
)


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


RELEVANT_KEYWORDS_PROMPT = """You must generate the relevant keywords to search for based on the context.

# Context
{{ context }}
"""


async def get_file_contents(
    context: str,
    toolkit: PyAIToolkit,
) -> str:
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
