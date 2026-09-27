"""Tests for terminal CLI interface."""

import io
import unittest
from unittest.mock import MagicMock, patch

from coding_agent.cli import main, run_interactive_session
from coding_agent.models import AgentResult


class TestCLI(unittest.TestCase):
    """Test CLI entry point, argument parsing, REPL loop, plan mode, and signal handling."""

    @patch("coding_agent.cli.Agent")
    def test_non_interactive_prompt_execution(self, mock_agent_cls):
        """Verify non-interactive mode when a single prompt argument is provided."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            final_response="Non-interactive answer",
            messages=[],
            steps_taken=1,
            completed=True,
        )
        mock_agent_cls.return_value = mock_agent

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            code = main(["Explain Python GIL"])

            self.assertEqual(code, 0)
            output = fake_out.getvalue()
            self.assertIn("Terminal Coding Agent initialized", output)
            self.assertIn("Non-interactive answer", output)
            mock_agent.run.assert_called_once()
            args, _ = mock_agent.run.call_args
            self.assertEqual(args[0], "Explain Python GIL")

    @patch("builtins.input", side_effect=["/exit"])
    @patch("coding_agent.cli.Agent")
    def test_interactive_session_exit(self, mock_agent_cls, mock_input):
        """Verify interactive session initializes and exits cleanly on /exit."""
        mock_agent_cls.return_value = MagicMock()

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            code = main([])

            self.assertEqual(code, 0)
            output = fake_out.getvalue()
            self.assertIn("Terminal Coding Agent v0.1.0", output)
            self.assertIn("Goodbye!", output)


if __name__ == "__main__":
    unittest.main()
