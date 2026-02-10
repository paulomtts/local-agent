"""Tests for read_files_tool_impl and its utility functions."""

from unittest.mock import AsyncMock, MagicMock
import pytest

from app.logic.read_files import (
    read_files_tool_impl,
    GenerateRelevantKeywords,
)
from app.utils import (
    search_files_by_name,
    search_files_by_content,
    read_file_contents,
    deduplicate_paths,
)


# ---------------------------------------------------------------------------
# search_files_by_name
# ---------------------------------------------------------------------------

class TestSearchFilesByName:
    def test_finds_utils_by_keyword(self):
        paths = search_files_by_name(["utils"])
        normalized = [p.replace("\\", "/") for p in paths]
        assert any("app/utils.py" in p for p in normalized)

    def test_excludes_venv(self):
        paths = search_files_by_name(["utils"])
        assert not any(".venv" in p for p in paths)

    def test_no_match_returns_empty(self):
        paths = search_files_by_name(["zzz_nonexistent_zzz"])
        assert paths == []


# ---------------------------------------------------------------------------
# search_files_by_content
# ---------------------------------------------------------------------------

class TestSearchFilesByContent:
    def test_finds_file_containing_keyword(self):
        paths = search_files_by_content(["ToolRegistry"])
        normalized = [p.replace("\\", "/") for p in paths]
        assert any("app/utils.py" in p for p in normalized)

    def test_excludes_venv(self):
        paths = search_files_by_content(["ToolRegistry"])
        assert not any(".venv" in p for p in paths)

    def test_no_match_returns_empty(self, tmp_path):
        (tmp_path / "example.py").write_text("hello world")
        paths = search_files_by_content(["nonexistent"], root=tmp_path)
        assert paths == []


# ---------------------------------------------------------------------------
# read_file_contents
# ---------------------------------------------------------------------------

class TestReadFileContents:
    def test_reads_file_with_header(self):
        result = read_file_contents(["app/utils.py"])
        assert "=== app/utils.py ===" in result
        assert "get_tools_definitions" in result

    def test_multiple_files(self):
        result = read_file_contents(["app/utils.py", "app/logic/think.py"])
        assert "=== app/utils.py ===" in result
        assert "=== app/logic/think.py ===" in result

    def test_missing_file_skipped(self):
        result = read_file_contents(["nonexistent.py", "app/utils.py"])
        assert "nonexistent" not in result
        assert "=== app/utils.py ===" in result

    def test_empty_list_returns_empty(self):
        result = read_file_contents([])
        assert result == ""


# ---------------------------------------------------------------------------
# deduplicate_paths
# ---------------------------------------------------------------------------

class TestDeduplicatePaths:
    def test_removes_duplicates(self):
        result = deduplicate_paths(["app/utils.py", "app/utils.py"])
        assert result == ["app/utils.py"]

    def test_normalizes_dot_slash(self):
        result = deduplicate_paths(["./app/utils.py", "app/utils.py"])
        assert result == ["app/utils.py"]

    def test_preserves_order(self):
        result = deduplicate_paths(["b.py", "a.py", "b.py"])
        assert result == ["b.py", "a.py"]


# ---------------------------------------------------------------------------
# read_files_tool_impl (with mocked LLM)
# ---------------------------------------------------------------------------

class TestReadFilesToolImpl:
    @pytest.mark.asyncio
    async def test_finds_and_reads_utils(self):
        mock_toolkit = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = GenerateRelevantKeywords(keywords=["utils"])
        mock_toolkit.asend.return_value = mock_response

        result = await read_files_tool_impl(
            context="Read the utils file",
            toolkit=mock_toolkit,
        )
        assert "=== app/utils.py ===" in result
        assert "get_tools_definitions" in result

    @pytest.mark.asyncio
    async def test_empty_keywords_after_strip(self):
        mock_toolkit = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = MagicMock()
        mock_response.content.keywords = ["  ", ""]
        mock_toolkit.asend.return_value = mock_response

        result = await read_files_tool_impl(
            context="something",
            toolkit=mock_toolkit,
        )
        assert result == "No relevant keywords found."

    @pytest.mark.asyncio
    async def test_no_files_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "empty.py").write_text("nothing useful here")

        mock_toolkit = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = GenerateRelevantKeywords(
            keywords=["nonexistent"]
        )
        mock_toolkit.asend.return_value = mock_response

        result = await read_files_tool_impl(
            context="something",
            toolkit=mock_toolkit,
        )
        assert result == "No relevant files found."

    @pytest.mark.asyncio
    async def test_finds_file_by_content_not_name(self):
        """A keyword in file content but not filename should still find the file."""
        mock_toolkit = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = GenerateRelevantKeywords(
            keywords=["ToolRegistry"]
        )
        mock_toolkit.asend.return_value = mock_response

        result = await read_files_tool_impl(
            context="Find ToolRegistry",
            toolkit=mock_toolkit,
        )
        assert "=== app/utils.py ===" in result
