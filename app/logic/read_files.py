import shlex

from pydantic import BaseModel, Field
from py_ai_toolkit import PyAIToolkit

from app.sessions import TerminalSession
from app.utils import (
    EXCLUSION_DIRS,
    EXCLUSION_FILES,
    clean_terminal_output,
    deduplicate_paths,
    extract_paths_from_grep_output,
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


async def read_files_tool_impl(
    terminal_session: TerminalSession,
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

    exclude_dirs_expressions = " ".join(
        f"--exclude-dir='{d}'" for d in EXCLUSION_DIRS
    )
    exclude_files_expressions = " ".join(
        f"--exclude='{f}'" for f in EXCLUSION_FILES
    )
    grep_output: list[str] = []

    for keyword in keywords:
        command = (
            "LC_ALL=C grep -r --line-number --color=never --binary-files=without-match "
            "--include='*.py' "
            f"{exclude_dirs_expressions} {exclude_files_expressions} "
            f"{shlex.quote(keyword)} ."
        )
        result = terminal_session.execute(command)
        if result:
            cleaned_result = clean_terminal_output(result)
            if cleaned_result:
                grep_output.append(cleaned_result)

    file_names = search_files_by_name(keywords)
    file_names += extract_paths_from_grep_output(grep_output)
    file_names = deduplicate_paths(file_names)

    if not file_names:
        return "No relevant files found."

    command = 'awk \'FNR==1{print "=== " FILENAME " ==="}{print}\' ' + " ".join(
        shlex.quote(name) for name in file_names
    )
    read_output = terminal_session.execute(command, timeout=10.0)
    return read_output or "No content read from files."
