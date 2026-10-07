"""Tests for terminal CLI interface."""

import io
import unittest
from unittest.mock import MagicMock, patch

from coding_agent.cli import main, run_interactive_session
from coding_agent.models import AgentResult
from coding_agent.pipeline import ExperiencePipeline
from coding_agent.run_record import RunOutcome, RunRecord


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
    @patch("builtins.input", side_effect=[KeyboardInterrupt, EOFError])
    def test_signal_handling(self, mock_input):
        """Verify KeyboardInterrupt is handled without crash and EOFError exits cleanly."""
        mock_agent = MagicMock()

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            code = run_interactive_session(mock_agent)

            self.assertEqual(code, 0)
            output = fake_out.getvalue()
            self.assertIn("KeyboardInterrupt (type /exit or press Ctrl-D to quit)", output)
            self.assertIn("Goodbye!", output)


class TestCLIPipelineIntegration(unittest.TestCase):
    """Tests for pipeline integration in the CLI."""

    def _make_run_record(self) -> RunRecord:
        record = RunRecord()
        record.record_task_start("Say hello")
        record.record_final_response("Hello!")
        record.final_response = "Hello!"
        record.steps_taken = 1
        record.outcome = RunOutcome.SUCCESS
        return record

    def test_pipeline_called_after_run_interactive(self):
        """Verify pipeline.process is called once per user turn in interactive mode."""
        record = self._make_run_record()
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            final_response="Hello!",
            messages=[],
            steps_taken=1,
            completed=True,
            run_record=record,
        )

        mock_pipeline = MagicMock(spec=ExperiencePipeline)

        with patch("builtins.input", side_effect=["hello", "/exit"]):
            with patch("sys.stdout", new=io.StringIO()):
                run_interactive_session(mock_agent, pipeline=mock_pipeline)

        mock_pipeline.process.assert_called_once()

    def test_pipeline_not_called_when_none(self):
        """Verify no crash when pipeline=None (backward compat)."""
        record = self._make_run_record()
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            final_response="Hello!",
            messages=[],
            steps_taken=1,
            completed=True,
            run_record=record,
        )

        with patch("builtins.input", side_effect=["hello", "/exit"]):
            with patch("sys.stdout", new=io.StringIO()):
                code = run_interactive_session(mock_agent, pipeline=None)

        self.assertEqual(code, 0)

    def test_pipeline_error_does_not_crash_cli(self):
        """Verify that a pipeline failure produces a warning but does not crash the REPL."""
        record = self._make_run_record()
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            final_response="Done",
            messages=[],
            steps_taken=1,
            completed=True,
            run_record=record,
        )

        mock_pipeline = MagicMock(spec=ExperiencePipeline)
        mock_pipeline.process.side_effect = RuntimeError("Storage unavailable")

        with patch("builtins.input", side_effect=["hello", "/exit"]):
            with patch("sys.stdout", new=io.StringIO()):
                with patch("sys.stderr", new=io.StringIO()) as err_out:
                    code = run_interactive_session(mock_agent, pipeline=mock_pipeline)

        self.assertEqual(code, 0)
        self.assertIn("Post-run pipeline failed", err_out.getvalue())

    @patch("coding_agent.cli.Agent")
    @patch("coding_agent.cli.ExperiencePipeline")
    @patch("coding_agent.cli.ExperienceMemory")
    def test_no_memory_flag_disables_memory(self, mock_mem_cls, mock_pipeline_cls, mock_agent_cls):
        """Verify --no-memory passes None memory to Agent and pipeline."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            final_response="Done", messages=[], steps_taken=1, completed=True
        )
        mock_agent_cls.return_value = mock_agent
        mock_pipeline_cls.return_value = MagicMock(spec=ExperiencePipeline)

        with patch("sys.stdout", new=io.StringIO()):
            main(["Say hello", "--no-memory"])

        # memory=None is passed to Agent when --no-memory is used
        agent_kwargs = mock_agent_cls.call_args[1]
        self.assertIsNone(agent_kwargs.get("memory"))

    @patch("coding_agent.cli.Agent")
    @patch("coding_agent.cli.ExperiencePipeline")
    @patch("coding_agent.cli.ExperienceMemory")
    def test_no_reflection_flag_passed_to_pipeline(self, mock_mem_cls, mock_pipeline_cls, mock_agent_cls):
        """Verify --no-reflection sets enable_reflection=False in pipeline."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            final_response="Done", messages=[], steps_taken=1, completed=True
        )
        mock_agent_cls.return_value = mock_agent
        mock_pipeline_cls.return_value = MagicMock(spec=ExperiencePipeline)

        with patch("sys.stdout", new=io.StringIO()):
            main(["Say hello", "--no-reflection"])

        pipeline_kwargs = mock_pipeline_cls.call_args[1]
        self.assertFalse(pipeline_kwargs.get("enable_reflection", True))

    @patch("coding_agent.cli.Agent")
    @patch("coding_agent.cli.ExperiencePipeline")
    @patch("coding_agent.cli.ExperienceMemory")
    def test_only_store_successful_flag_passed_to_pipeline(
        self, mock_mem_cls, mock_pipeline_cls, mock_agent_cls
    ):
        """Verify --only-store-successful is forwarded to the pipeline."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            final_response="Done", messages=[], steps_taken=1, completed=True
        )
        mock_agent_cls.return_value = mock_agent
        mock_pipeline_cls.return_value = MagicMock(spec=ExperiencePipeline)

        with patch("sys.stdout", new=io.StringIO()):
            main(["Say hello", "--only-store-successful"])

        pipeline_kwargs = mock_pipeline_cls.call_args[1]
        self.assertTrue(pipeline_kwargs.get("only_store_successful", False))


if __name__ == "__main__":
    unittest.main()
