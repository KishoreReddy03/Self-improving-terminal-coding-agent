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

    @patch("builtins.input", side_effect=["/plan", "Do work", "/exit"])
    def test_plan_mode_toggle_in_interactive_session(self, mock_input):
        """Verify toggling plan mode updates prompt status and appends directive."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            final_response="Planned response",
            messages=[{"role": "user", "content": "req"}, {"role": "assistant", "content": "res"}],
            steps_taken=1,
            completed=True,
        )

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            code = run_interactive_session(mock_agent)

            self.assertEqual(code, 0)
            output = fake_out.getvalue()
            self.assertIn("Plan mode is now ENABLED", output)
            self.assertIn("Planned response", output)

            call_args = mock_agent.run.call_args[0][0]
            self.assertIn("[PLAN MODE INSTRUCTION]", call_args[0]["content"])

    @patch("builtins.input", side_effect=["first prompt", "/clear", "second prompt", "/exit"])
    def test_clear_conversation_history(self, mock_input):
        """Verify /clear resets accumulated conversation history."""
        mock_agent = MagicMock()
        mock_agent.run.side_effect = [
            AgentResult(final_response="ans 1", messages=[{"role": "user", "content": "1"}, {"role": "assistant", "content": "ans 1"}]),
            AgentResult(final_response="ans 2", messages=[{"role": "user", "content": "2"}, {"role": "assistant", "content": "ans 2"}]),
        ]

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            code = run_interactive_session(mock_agent)

            self.assertEqual(code, 0)
            output = fake_out.getvalue()
            self.assertIn("Conversation history cleared", output)
            self.assertEqual(mock_agent.run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
