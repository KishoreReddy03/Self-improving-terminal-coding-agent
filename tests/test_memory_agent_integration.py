"""Tests for connecting experience retrieval to new agent runs."""

import tempfile
from unittest.mock import MagicMock

import pytest

from coding_agent.agent import Agent
from coding_agent.evaluator import EvaluationResult
from coding_agent.memory import Experience, ExperienceMemory
from coding_agent.models import AgentConfig, ModelResponse
from coding_agent.reflection import ReflectionResult


@pytest.fixture
def memory_with_experiences():
    with tempfile.TemporaryDirectory() as tmp_dir:
        memory = ExperienceMemory(storage_dir=tmp_dir)

        # Relevant experience for database tasks
        memory.add(
            task="Optimize SQLite database connection pooling",
            evaluation=EvaluationResult(
                success=True, score=1.0, reason="Clean run", details={}
            ),
            reflection=ReflectionResult(
                what_worked="Used PRAGMA journal_mode=WAL to speed up transactions.",
                what_failed="None",
                why_it_failed="N/A",
                what_to_do_differently="Set timeout parameter on connection.",
            ),
        )

        # Irrelevant experience for UI layout tasks
        memory.add(
            task="Create responsive CSS flexbox navbar navigation header",
            evaluation=EvaluationResult(
                success=True, score=1.0, reason="Clean run", details={}
            ),
            reflection=ReflectionResult(
                what_worked="Used display flex and gap utilities.",
                what_failed="None",
                why_it_failed="N/A",
                what_to_do_differently="Test on mobile breakpoint.",
            ),
        )

        yield memory


class TestAgentMemoryIntegration:
    def test_agent_retrieves_relevant_experience_context(self, memory_with_experiences):
        mock_client = MagicMock()
        mock_client.config = AgentConfig()
        mock_client.generate_response.return_value = ModelResponse(
            content="Database connection updated."
        )

        agent = Agent(client=mock_client, memory=memory_with_experiences)
        result = agent.run("Fix SQLite database timeout issue")

        # Inspect the messages sent to generate_response
        sent_messages = mock_client.generate_response.call_args[1]["messages"]

        # Verify optional context message was inserted
        context_msgs = [
            m for m in sent_messages if "[PAST EXPERIENCE ADVICE" in m.get("content", "")
        ]
        assert len(context_msgs) == 1
        advice_content = context_msgs[0]["content"]
        assert "PRAGMA journal_mode=WAL" in advice_content
        assert "Set timeout parameter on connection" in advice_content
        assert "CSS flexbox" not in advice_content

    def test_agent_ignores_irrelevant_experience(self, memory_with_experiences):
        mock_client = MagicMock()
        mock_client.config = AgentConfig()
        mock_client.generate_response.return_value = ModelResponse(
            content="Parsing complete."
        )

        agent = Agent(client=mock_client, memory=memory_with_experiences)
        # Query task unrelated to database or CSS navbar
        result = agent.run("Parse abstract syntax tree for Python functions")

        sent_messages = mock_client.generate_response.call_args[1]["messages"]
        context_msgs = [
            m for m in sent_messages if "[PAST EXPERIENCE ADVICE" in m.get("content", "")
        ]
        assert len(context_msgs) == 0

    def test_agent_runs_normally_when_memory_is_none(self):
        mock_client = MagicMock()
        mock_client.config = AgentConfig()
        mock_client.generate_response.return_value = ModelResponse(
            content="Task finished."
        )

        agent = Agent(client=mock_client, memory=None)
        result = agent.run("Write unit test for calculator")

        # Verify no optional memory context system message was added
        system_msgs = [
            m for m in result.messages if "[PAST EXPERIENCE ADVICE" in m.get("content", "")
        ]
        assert len(system_msgs) == 0
        assert result.messages[0]["role"] == "user"
        assert result.messages[0]["content"] == "Write unit test for calculator"
