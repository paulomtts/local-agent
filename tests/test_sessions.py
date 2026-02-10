"""Tests for TerminalSession.execute — the core integration point.

These are integration tests that create a real PTY to verify that
execute() returns clean command output without prompt/echo noise.
"""

from app.factories import get_terminal_session


class TestTerminalSessionExecute:
    """Verify that execute() returns only command output."""

    def setup_method(self):
        self.session = get_terminal_session()

    def teardown_method(self):
        self.session.close()

    def test_simple_echo(self):
        result = self.session.execute("echo hello")
        assert result == "hello"

    def test_no_prompt_in_output(self):
        result = self.session.execute("echo hello")
        assert "$" not in result
        assert "bash" not in result.lower()

    def test_no_echo_of_command(self):
        result = self.session.execute("echo hello")
        # Should not contain the command itself, just its output
        assert "echo" not in result

    def test_multiline_output(self):
        result = self.session.execute("printf 'line1\nline2\nline3'")
        lines = result.split("\n")
        assert lines == ["line1", "line2", "line3"]

    def test_empty_output_command(self):
        result = self.session.execute("true")
        assert result == ""

    def test_consecutive_commands_no_leakage(self):
        """Output from one command must not leak into the next."""
        r1 = self.session.execute("echo first")
        r2 = self.session.execute("echo second")
        assert r1 == "first"
        assert r2 == "second"
        assert "first" not in r2
        assert "second" not in r1

    def test_grep_command(self):
        """grep is the primary use case — verify it works."""
        result = self.session.execute(
            "echo 'hello world' | grep hello"
        )
        assert "hello world" in result

    def test_grep_no_match_returns_empty(self):
        result = self.session.execute(
            "echo 'hello' | grep nonexistent || true"
        )
        # grep returns exit 1 on no match; || true prevents error
        # Output should be empty or just whitespace
        assert "nonexistent" not in result

    def test_awk_command(self):
        """awk is used to read files — verify single-quote handling."""
        result = self.session.execute(
            "echo 'test content' | awk '{print $0}'"
        )
        assert result == "test content"

    def test_awk_with_filename_header(self):
        """Reproduce the exact awk pattern used in read_files.py."""
        # Create a temp file, read it with awk
        self.session.execute("echo 'line1' > /tmp/test_rf.txt")
        result = self.session.execute(
            "awk 'FNR==1{print \"=== \" FILENAME \" ===\"}{print}' /tmp/test_rf.txt"
        )
        assert "=== /tmp/test_rf.txt ===" in result
        assert "line1" in result

    def test_command_with_special_chars(self):
        """Commands with quotes, pipes, etc. should work."""
        result = self.session.execute("echo 'it'\\''s a test'")
        assert "it's a test" in result

    def test_long_output(self):
        """Verify we don't truncate or lose long output."""
        result = self.session.execute("seq 1 100")
        lines = result.strip().split("\n")
        assert len(lines) == 100
        assert lines[0] == "1"
        assert lines[-1] == "100"

    def test_marker_not_in_output(self):
        """The internal markers must never appear in returned output."""
        result = self.session.execute("echo hello")
        assert "__CMD_" not in result

    def test_stderr_captured(self):
        """PTY merges stderr into stdout — verify we capture it."""
        result = self.session.execute("echo error >&2")
        assert "error" in result

    def test_real_grep_on_codebase(self):
        """Run the actual grep pattern from read_files.py against this repo."""
        result = self.session.execute(
            "LC_ALL=C grep -r --line-number --color=never "
            "--binary-files=without-match --include='*.py' "
            "--exclude-dir='.venv' --exclude-dir='.git' "
            "TerminalSession ."
        )
        assert "app/sessions.py" in result
        assert "class TerminalSession" in result

    def test_real_awk_on_codebase(self):
        """Run the actual awk pattern from read_files.py on a known file."""
        result = self.session.execute(
            "awk 'FNR==1{print \"=== \" FILENAME \" ===\"}{print}' ./app/utils.py",
            timeout=10.0,
        )
        assert "=== ./app/utils.py ===" in result
        assert "get_tools_definitions" in result
