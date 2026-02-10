import re
from pathlib import Path

from pygents import ToolRegistry


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
CONTROL_RE = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")

EXCLUSION_DIRS = [
    ".venv",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".cursor",
    ".vscode",
]

EXCLUSION_FILES = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.pyw",
    "*.pyz",
    ".gitignore",
    ".env",
    ".env.*",
]


def get_tools_definitions() -> str:
    definitions = ""
    for tool in ToolRegistry.all():
        definitions += f"{tool.metadata.name}: {tool.metadata.description}\n"
    return definitions


def clean_terminal_output(raw_output: str) -> str:
    cleaned_output = ANSI_ESCAPE_RE.sub("", raw_output)
    cleaned_output = CONTROL_RE.sub("", cleaned_output)
    return cleaned_output


def extract_paths_from_grep_output(grep_output: list[str]) -> list[str]:
    """Extract file paths from grep --line-number output lines."""
    paths: list[str] = []
    lines = "\n".join(grep_output).strip().split("\n")
    for line in lines:
        if not line or ":" not in line:
            continue
        path = line.split(":", 1)[0].strip()
        if path:
            paths.append(path)
    return paths


def search_files_by_name(
    keywords: list[str],
    exclusion_dirs: list[str] | None = None,
) -> list[str]:
    """Find .py files whose names contain any of the given keywords."""
    if exclusion_dirs is None:
        exclusion_dirs = EXCLUSION_DIRS
    paths: list[str] = []
    for keyword in keywords:
        for path in Path(".").rglob(f"*{keyword}*.py"):
            if any(excluded in path.parts for excluded in exclusion_dirs):
                continue
            paths.append(str(path))
    return paths


def deduplicate_paths(paths: list[str]) -> list[str]:
    """Normalize and deduplicate file paths, preserving order."""
    return list(dict.fromkeys(str(Path(name)) for name in paths))
