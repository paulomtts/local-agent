"""Tests for app/logic/read_files.py.

Unit tests that mock TerminalSession and PyAIToolkit to isolate
the search-and-read logic.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.logic.read_files import (
    read_files_tool_impl,
    GenerateRelevantKeywords,
)
from app.utils import (
    clean_terminal_output,
    extract_paths_from_grep_output,
    search_files_by_name,
    deduplicate_paths,
)


# ---------------------------------------------------------------------------
# clean_terminal_output
# ---------------------------------------------------------------------------

class TestCleanTerminalOutput:
    def test_strips_ansi_color_codes(self):
        raw = "\x1B[31mhello\x1B[0m"
        assert clean_terminal_output(raw) == "hello"

    def test_strips_cursor_movement(self):
        raw = "\x1B[2Jhello"
        assert clean_terminal_output(raw) == "hello"

    def test_strips_carriage_return(self):
        raw = "hello\r\nworld"
        assert clean_terminal_output(raw) == "hello\nworld"

    def test_strips_null_bytes(self):
        raw = "hello\x00world"
        assert clean_terminal_output(raw) == "helloworld"

    def test_preserves_newlines(self):
        raw = "hello world\nline2"
        assert clean_terminal_output(raw) == "hello world\nline2"

    def test_strips_bracketed_paste(self):
        raw = "\x1B[?2004hhello\x1B[?2004l"
        assert clean_terminal_output(raw) == "hello"


# ---------------------------------------------------------------------------
# read_files_tool_impl — grep output parsing
# ---------------------------------------------------------------------------

class TestGrepOutputParsing:
    """Test that file paths are correctly extracted from grep output."""

    @pytest.mark.asyncio
    async def test_extracts_paths_from_grep(self):
        mock_session = MagicMock()
        mock_toolkit = AsyncMock()

        # Simulate LLM returning keywords
        mock_response = MagicMock()
        mock_response.content = GenerateRelevantKeywords(
            keywords=["TerminalSession"]
        )
        mock_toolkit.asend.return_value = mock_response

        # Simulate grep finding a match
        mock_session.execute.side_effect = [
            # grep call
            "./app/sessions.py:8:class TerminalSession:\n"
            "./app/factories.py:3:from app.sessions import TerminalSession",
            # awk call
            "=== ./app/sessions.py ===\nimport os\n=== ./app/factories.py ===\nimport pty",
        ]

        result = await read_files_tool_impl(
            terminal_session=mock_session,
            context="Find the TerminalSession class",
            toolkit=mock_toolkit,
        )

        assert "=== ./app/sessions.py ===" in result
        assert "=== ./app/factories.py ===" in result

    @pytest.mark.asyncio
    async def test_empty_keywords_after_strip_returns_message(self):
        """If LLM returns only whitespace keywords, they get stripped to empty."""
        mock_session = MagicMock()
        mock_toolkit = AsyncMock()

        mock_response = MagicMock()
        mock_response.content = MagicMock()
        mock_response.content.keywords = ["  ", ""]
        mock_toolkit.asend.return_value = mock_response

        result = await read_files_tool_impl(
            terminal_session=mock_session,
            context="something",
            toolkit=mock_toolkit,
        )
        assert result == "No relevant keywords found."

    @pytest.mark.asyncio
    async def test_no_grep_results_returns_message(self):
        mock_session = MagicMock()
        mock_toolkit = AsyncMock()

        mock_response = MagicMock()
        mock_response.content = GenerateRelevantKeywords(
            keywords=["nonexistent_keyword"]
        )
        mock_toolkit.asend.return_value = mock_response

        # grep returns nothing
        mock_session.execute.return_value = ""

        result = await read_files_tool_impl(
            terminal_session=mock_session,
            context="something",
            toolkit=mock_toolkit,
        )
        assert result == "No relevant files found."

    @pytest.mark.asyncio
    async def test_awk_empty_returns_no_content_message(self):
        mock_session = MagicMock()
        mock_toolkit = AsyncMock()

        mock_response = MagicMock()
        mock_response.content = GenerateRelevantKeywords(keywords=["utils"])
        mock_toolkit.asend.return_value = mock_response

        # grep finds a file, awk returns nothing
        mock_session.execute.side_effect = [
            "./app/utils.py:1:from pygents import ToolRegistry",  # grep
            "",  # awk
        ]

        result = await read_files_tool_impl(
            terminal_session=mock_session,
            context="read utils",
            toolkit=mock_toolkit,
        )
        assert result == "No content read from files."

    @pytest.mark.asyncio
    async def test_deduplicates_file_names(self):
        mock_session = MagicMock()
        mock_toolkit = AsyncMock()

        mock_response = MagicMock()
        mock_response.content = GenerateRelevantKeywords(
            keywords=["utils", "util"]
        )
        mock_toolkit.asend.return_value = mock_response

        calls = []

        def track_execute(cmd, **kwargs):
            calls.append(cmd)
            if "grep" in cmd:
                return "./app/utils.py:1:from pygents import ToolRegistry"
            if "awk" in cmd:
                return "=== ./app/utils.py ===\nfrom pygents import ToolRegistry"
            return ""

        mock_session.execute.side_effect = track_execute

        result = await read_files_tool_impl(
            terminal_session=mock_session,
            context="read utils",
            toolkit=mock_toolkit,
        )

        # Verify awk was called with the file only once (deduplicated)
        awk_calls = [c for c in calls if "awk" in c]
        assert len(awk_calls) == 1
        assert awk_calls[0].count("app/utils.py") == 1

    @pytest.mark.asyncio
    async def test_finds_file_by_name_even_when_grep_misses_content(self):
        """Keyword 'utils' should find app/utils.py by filename even if
        grep doesn't match its contents."""
        mock_session = MagicMock()
        mock_toolkit = AsyncMock()

        mock_response = MagicMock()
        mock_response.content = GenerateRelevantKeywords(keywords=["utils"])
        mock_toolkit.asend.return_value = mock_response

        calls = []

        def track_execute(cmd, **kwargs):
            calls.append(cmd)
            if "grep" in cmd:
                return ""  # grep finds nothing in content
            if "awk" in cmd:
                return "=== app/utils.py ===\nfrom pygents import ToolRegistry"
            return ""

        mock_session.execute.side_effect = track_execute

        result = await read_files_tool_impl(
            terminal_session=mock_session,
            context="read utils",
            toolkit=mock_toolkit,
        )

        # File should be found by name and read via awk
        awk_calls = [c for c in calls if "awk" in c]
        assert len(awk_calls) == 1
        assert "utils" in awk_calls[0]
        assert "=== app/utils.py ===" in result

    @pytest.mark.asyncio
    async def test_handles_grep_lines_without_colon(self):
        """Lines without ':' (like blank lines or warnings) should be skipped."""
        mock_session = MagicMock()
        mock_toolkit = AsyncMock()

        mock_response = MagicMock()
        mock_response.content = GenerateRelevantKeywords(keywords=["test"])
        mock_toolkit.asend.return_value = mock_response

        mock_session.execute.side_effect = [
            # grep returns some noise alongside a valid match
            "Binary file matches\n./app/utils.py:1:test line",
            "=== ./app/utils.py ===\ntest line",
        ]

        result = await read_files_tool_impl(
            terminal_session=mock_session,
            context="find test",
            toolkit=mock_toolkit,
        )
        assert "=== ./app/utils.py ===" in result
