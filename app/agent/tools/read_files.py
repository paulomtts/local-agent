import uuid
from pathlib import Path

from py_ai_toolkit import PyAIToolkit
from pydantic import BaseModel, Field
from pygents import ContextItem, ContextQueue, Turn, tool

from app.agent.tools.think import think
from app.core.factories import get_toolkit
from app.core.logger import log_token_usage, log_tool_use
from app.memory import ToolCall, get_user_messages, write_episodic_event

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXCLUSION_DIRS = [
    ".venv",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
]

EXCLUSION_EXTENSIONS = {".lock", ".pyc", ".pyo", ".env"}


# ---------------------------------------------------------------------------
# File-system utilities
# ---------------------------------------------------------------------------


def get_file_tree(root: str = ".") -> str:
    """List all non-excluded source files under root, one path per line."""
    paths = sorted(
        str(p)
        for p in Path(root).rglob("*")
        if p.is_file()
        and not any(ex in p.parts for ex in EXCLUSION_DIRS)
        and p.suffix not in EXCLUSION_EXTENSIONS
    )
    return "\n".join(paths) or "(empty)"


def read_file_contents(file_paths: list[str]) -> str:
    """Read files and format with === path === headers."""
    parts: list[str] = []
    for path in file_paths:
        try:
            content = Path(path).read_text()
            parts.append(f"=== {path} ===\n{content}")
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

RELEVANT_FILES_PROMPT = """Select the files from the project file tree that are relevant to the user's request. Return their exact paths as listed in the tree.

# Project File Tree
{{ file_tree }}

# Working Memory
{{ user_messages }}
"""


class SelectRelevantFiles(BaseModel):
    "Select relevant file paths from the project file tree."

    paths: list[str] = Field(
        default_factory=list,
        description="Exact file paths from the project file tree that are relevant to the user's request.",
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


async def get_file_contents(
    memory: ContextQueue,
    toolkit: PyAIToolkit,
) -> str:
    user_messages = get_user_messages(memory, n=5)
    file_tree = get_file_tree()

    result = await toolkit.asend(
        response_model=SelectRelevantFiles,
        template=RELEVANT_FILES_PROMPT,
        file_tree=file_tree,
        user_messages=user_messages,
    )

    log_token_usage("read_files", result)
    if not isinstance(result.content, SelectRelevantFiles):
        raise ValueError("Expected SelectRelevantFiles, got %s" % type(result.content))

    tree_paths = set(file_tree.splitlines())
    valid_paths = [p for p in result.content.paths if p in tree_paths]

    if not valid_paths:
        return "No relevant files found."

    return read_file_contents(valid_paths) or "No content read from files."


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool()
async def read_files(memory: ContextQueue):
    "Use to find and read relevant files."
    log_tool_use("read_files")
    toolkit = get_toolkit()
    file_contents = await get_file_contents(
        memory=memory,
        toolkit=toolkit,
    )

    lines = file_contents.split("\n")
    file_mentions = [
        line for line in lines if line.startswith("=== ") and line.endswith(" ===")
    ]
    file_count = len(file_mentions)

    if file_count > 0:
        file_names = [
            mention[4:-4].strip().split("/")[-1] for mention in file_mentions[:3]
        ]

        write_episodic_event(
            event=f"agent read {file_count} file{'s' if file_count > 1 else ''}",
            context=", ".join(file_names) if file_names else None,
        )

        description = f"Files: {file_count} file(s) read ({', '.join(file_names[:3])})"
    else:
        description = "No relevant files found."

    item_id = f"files_{uuid.uuid4().hex[:8]}"
    tool_call = ToolCall(
        tool_name="read_files", result=f"[pool:{item_id}] {description}"
    )
    yield ContextItem[ToolCall](tool_call)
    yield ContextItem[str](content=file_contents, description=description, id=item_id)
    yield Turn(think)
