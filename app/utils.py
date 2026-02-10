from pathlib import Path

from pygents import ToolRegistry


EXCLUSION_DIRS = [
    ".venv",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".cursor",
    ".vscode",
]


def get_tools_definitions() -> str:
    definitions = ""
    for tool in ToolRegistry.all():
        definitions += f"{tool.metadata.name}: {tool.metadata.description}\n"
    return definitions


def search_files_by_name(
    keywords: list[str],
    root: Path | str = ".",
    exclusion_dirs: list[str] | None = None,
) -> list[str]:
    """Find .py files whose names contain any of the given keywords."""
    if exclusion_dirs is None:
        exclusion_dirs = EXCLUSION_DIRS
    paths: list[str] = []
    for keyword in keywords:
        for path in Path(root).rglob(f"*{keyword}*.py"):
            if any(excluded in path.parts for excluded in exclusion_dirs):
                continue
            paths.append(str(path))
    return paths


def search_files_by_content(
    keywords: list[str],
    root: Path | str = ".",
    exclusion_dirs: list[str] | None = None,
) -> list[str]:
    """Find .py files whose contents contain any of the given keywords."""
    if exclusion_dirs is None:
        exclusion_dirs = EXCLUSION_DIRS
    paths: list[str] = []
    for py_file in Path(root).rglob("*.py"):
        if any(excluded in py_file.parts for excluded in exclusion_dirs):
            continue
        try:
            content = py_file.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for keyword in keywords:
            if keyword in content:
                paths.append(str(py_file))
                break
    return paths


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


def deduplicate_paths(paths: list[str]) -> list[str]:
    """Normalize and deduplicate file paths, preserving order."""
    return list(dict.fromkeys(str(Path(name)) for name in paths))
