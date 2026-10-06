"""Unit tests for ReflectionResult, ReflectionGenerator, and reflection storage integration."""

import json
import tempfile
from unittest.mock import MagicMock

import pytest

from coding_agent.client import LLMClient
from coding_agent.evaluator import EvaluationResult, TrajectoryEvaluator
from coding_agent.models import AgentConfig
from coding_agent.reflection import ReflectionGenerator, ReflectionResult
from coding_agent.run_record import RunEvent, RunEventType, RunOutcome, RunRecord
from coding_agent.trajectory_store import TrajectoryStore


class TestReflectionResult:
    def test_to_dict_and_from_dict(self):
        result = ReflectionResult(
            what_worked="File reading succeeded.",
            what_failed="Syntax error in test file.",
            why_it_failed="Missing parenthesis on line 4.",
            what_to_do_differently="Check syntax before saving file.",
            summary="Partial success.",
            raw_response="raw text",
        )
        d = result.to_dict()
        assert d["what_worked"] == "File reading succeeded."
        assert d["what_failed"] == "Syntax error in test file."
        assert d["why_it_failed"] == "Missing parenthesis on line 4."
        assert d["what_to_do_differently"] == "Check syntax before saving file."

        reconstructed = ReflectionResult.from_dict(d)
        assert reconstructed == result

    def test_json_round_trip(self):
        result = ReflectionResult(
            what_worked="All tests passed.",
            what_failed="None.",
            why_it_failed="N/A",
            what_to_do_differently="Maintain existing pattern.",
            summary="Clean execution.",
        )
        json_str = result.to_json(indent=2)
        parsed = json.loads(json_str)
        assert parsed["what_worked"] == "All tests passed."

        reconstructed = ReflectionResult.from_json(json_str)
        assert reconstructed == result


class TestReflectionGeneratorWithFakeModel:
    def make_fake_client(self, model_json_response: dict):
        """Create an LLMClient with a fake transport returning the given JSON dictionary."""
        def fake_transport(payload, headers, timeout):
            return {
                "id": "gen-123",
                "object": "chat.completion",
                "created": 12345678,
                "model": "fake-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(model_json_response),
                        },
                        "finish_reason": "stop",
                    }
                ],
            }

        return LLMClient(config=AgentConfig(), transport=fake_transport)

    @pytest.fixture
    def sample_record_and_eval(self):
        record = RunRecord(task="Fix bug in calculator")
        record.record_task_start("Fix bug in calculator")
        record.record_tool_call(
            step=1,
            tool_name="read_file",
            arguments={"path": "calc.py"},
            tool_call_id="call_1",
            approved=None,
        )
        record.record_tool_result(
            step=1,
            tool_name="read_file",
            tool_call_id="call_1",
            output="def add(a, b): return a + b",
            is_error=False,
        )
        record.record_final_response("Fixed calculator addition bug.")
        record.outcome = RunOutcome.SUCCESS
        record.steps_taken = 1

        evaluator = TrajectoryEvaluator()
        evaluation = evaluator.evaluate(record)

        return record, evaluation

    def test_generate_reflection_structured_json(self, sample_record_and_eval):
        record, evaluation = sample_record_and_eval
        fake_response = {
            "what_worked": "Read the calculator implementation accurately.",
            "what_failed": "None.",
            "why_it_failed": "N/A",
            "what_to_do_differently": "Keep using read_file before editing.",
            "summary": "Successful bugfix run.",
        }
        client = self.make_fake_client(fake_response)
        generator = ReflectionGenerator(client=client)

        reflection = generator.generate(record, evaluation)
        assert reflection.what_worked == "Read the calculator implementation accurately."
        assert reflection.what_failed == "None."
        assert reflection.why_it_failed == "N/A"
        assert reflection.what_to_do_differently == "Keep using read_file before editing."
        assert reflection.summary == "Successful bugfix run."

    def test_generate_reflection_markdown_code_block(self, sample_record_and_eval):
        record, evaluation = sample_record_and_eval
        fake_dict = {
            "what_worked": "Identified root cause.",
            "what_failed": "One tool error.",
            "why_it_failed": "Invalid parameter.",
            "what_to_do_differently": "Validate inputs.",
            "summary": "Completed with retries.",
        }
        markdown_content = f"```json\n{json.dumps(fake_dict)}\n```"

        def fake_transport(payload, headers, timeout):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": markdown_content,
                        }
                    }
                ]
            }

        client = LLMClient(config=AgentConfig(), transport=fake_transport)
        generator = ReflectionGenerator(client=client)

        reflection = generator.generate(record, evaluation)
        assert reflection.what_worked == "Identified root cause."
        assert reflection.what_failed == "One tool error."

    def test_generate_reflection_plain_text_fallback(self, sample_record_and_eval):
        record, evaluation = sample_record_and_eval
        plain_text = "The agent succeeded by analyzing the file directly."

        def fake_transport(payload, headers, timeout):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": plain_text,
                        }
                    }
                ]
            }

        client = LLMClient(config=AgentConfig(), transport=fake_transport)
        generator = ReflectionGenerator(client=client)

        reflection = generator.generate(record, evaluation)
        assert reflection.what_worked == plain_text
        assert reflection.raw_response == plain_text


class TestReflectionStorageIntegration:
    def test_save_and_retrieve_reflection_with_trajectory(self):
        record = RunRecord(task="Sample storage task")
        record.record_task_start("Sample storage task")
        record.outcome = RunOutcome.SUCCESS

        evaluation = EvaluationResult(
            success=True, score=1.0, reason="Clean run.", details={}
        )
        reflection = ReflectionResult(
            what_worked="Task completed.",
            what_failed="None",
            why_it_failed="N/A",
            what_to_do_differently="None",
            summary="All good.",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = TrajectoryStore(storage_dir=tmp_dir)
            filepath = store.save(record, evaluation=evaluation, reflection=reflection)

            assert filepath.is_file()

            retrieved_eval = store.get_evaluation(record.run_id)
            retrieved_refl = store.get_reflection(record.run_id)

            assert retrieved_eval == evaluation
            assert retrieved_refl == reflection
