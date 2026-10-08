"""Unit tests for TrajectoryEvaluator and EvaluationResult."""

import json

import pytest

from coding_agent.evaluator import EvaluationResult, TrajectoryEvaluator
from coding_agent.run_record import RunEvent, RunEventType, RunOutcome, RunRecord


class TestEvaluationResult:
    def test_to_dict_and_from_dict_round_trip(self):
        result = EvaluationResult(
            success=True,
            score=0.95,
            reason="Test completed cleanly.",
            details={"completed": True, "failed_tool_count": 0},
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["score"] == 0.95
        assert d["reason"] == "Test completed cleanly."

        reconstructed = EvaluationResult.from_dict(d)
        assert reconstructed == result

    def test_json_round_trip(self):
        result = EvaluationResult(
            success=False,
            score=0.35,
            reason="Tool execution failed.",
            details={"completed": True, "failed_tool_count": 1},
        )
        json_str = result.to_json(indent=2)
        parsed = json.loads(json_str)
        assert parsed["success"] is False

        reconstructed = EvaluationResult.from_json(json_str)
        assert reconstructed == result


class TestTrajectoryEvaluator:
    @pytest.fixture
    def evaluator(self):
        return TrajectoryEvaluator()

    def test_evaluate_successful_run_no_tools(self, evaluator):
        record = RunRecord(task="Greeting prompt")
        record.record_task_start("Greeting prompt")
        record.record_llm_call(step=1, message_count=1, has_tools=True)
        record.record_final_response("Hello! How can I help you today?")
        record.outcome = RunOutcome.SUCCESS
        record.steps_taken = 1

        eval_res = evaluator.evaluate(record)
        assert eval_res.success is True
        assert eval_res.score == 1.0
        assert "completed successfully" in eval_res.reason
        assert eval_res.details["completed"] is True
        assert eval_res.details["hit_max_steps"] is False
        assert eval_res.details["has_tool_failure"] is False

    def test_evaluate_successful_run_with_clean_tool_calls(self, evaluator):
        record = RunRecord(task="Read file task")
        record.record_task_start("Read file task")
        record.record_llm_call(step=1, message_count=1, has_tools=True)
        record.record_tool_call(
            step=1,
            tool_name="read_file",
            arguments={"path": "main.py"},
            tool_call_id="call_1",
            approved=None,
        )
        record.record_tool_result(
            step=1,
            tool_name="read_file",
            tool_call_id="call_1",
            output="print('hello')",
            is_error=False,
        )
        record.record_final_response("File content inspected.")
        record.outcome = RunOutcome.SUCCESS
        record.steps_taken = 1

        eval_res = evaluator.evaluate(record)
        assert eval_res.success is True
        assert eval_res.score == 1.0
        assert eval_res.details["total_tool_calls"] == 1
        assert eval_res.details["failed_tool_count"] == 0

    def test_evaluate_max_steps_reached(self, evaluator):
        record = RunRecord(task="Long task")
        record.record_task_start("Long task")
        record.outcome = RunOutcome.MAX_STEPS_REACHED
        record.steps_taken = 40

        eval_res = evaluator.evaluate(record)
        assert eval_res.success is False
        assert eval_res.score < 1.0
        assert eval_res.details["hit_max_steps"] is True
        assert "maximum step iteration limit" in eval_res.reason

    def test_evaluate_unhandled_error(self, evaluator):
        record = RunRecord(task="Failing task")
        record.record_task_start("Failing task")
        record.record_error(error_type="RuntimeError", message="API Connection failed")
        record.outcome = RunOutcome.ERROR
        record.error_message = "API Connection failed"
        record.steps_taken = 1

        eval_res = evaluator.evaluate(record)
        assert eval_res.success is False
        assert eval_res.details["has_error"] is True
        assert "API Connection failed" in eval_res.reason

    def test_evaluate_failed_tool_call(self, evaluator):
        record = RunRecord(task="Broken tool task")
        record.record_task_start("Broken tool task")
        record.record_llm_call(step=1, message_count=1, has_tools=True)
        record.record_tool_call(
            step=1,
            tool_name="read_file",
            arguments={"path": "nonexistent.txt"},
            tool_call_id="call_1",
            approved=None,
        )
        record.record_tool_result(
            step=1,
            tool_name="read_file",
            tool_call_id="call_1",
            output="Error: File 'nonexistent.txt' does not exist.",
            is_error=True,
        )
        record.record_final_response("File was not found.")
        record.outcome = RunOutcome.SUCCESS
        record.steps_taken = 1

        eval_res = evaluator.evaluate(record)
        assert eval_res.success is False
        assert eval_res.details["has_tool_failure"] is True
        assert eval_res.details["failed_tool_count"] == 1
        assert "1 of 1 tool call(s) failed" in eval_res.reason

    def test_evaluate_successful_verification_command(self, evaluator):
        record = RunRecord(task="Run test suite")
        record.record_task_start("Run test suite")
        record.record_tool_call(
            step=1,
            tool_name="shell_command",
            arguments={"command": "python -m pytest tests/"},
            tool_call_id="call_verif_1",
            approved=True,
        )
        record.record_tool_result(
            step=1,
            tool_name="shell_command",
            tool_call_id="call_verif_1",
            output=json.dumps({"exit_code": 0, "stdout": "10 passed", "stderr": ""}),
            is_error=False,
        )
        record.record_final_response("All tests passed cleanly.")
        record.outcome = RunOutcome.SUCCESS
        record.steps_taken = 1

        eval_res = evaluator.evaluate(record)
        assert eval_res.success is True
        assert eval_res.details["verification_run"] is True
        assert eval_res.details["verification_succeeded"] is True
        assert "Final verification command succeeded" in eval_res.reason

    def test_evaluate_failed_verification_command(self, evaluator):
        record = RunRecord(task="Fix broken bug")
        record.record_task_start("Fix broken bug")
        record.record_tool_call(
            step=1,
            tool_name="shell_command",
            arguments={"command": "pytest tests/test_agent.py"},
            tool_call_id="call_verif_2",
            approved=True,
        )
        record.record_tool_result(
            step=1,
            tool_name="shell_command",
            tool_call_id="call_verif_2",
            output=json.dumps({"exit_code": 1, "stdout": "1 failed", "stderr": ""}),
            is_error=True,
        )
        record.record_final_response("I attempted to fix the bug.")
        record.outcome = RunOutcome.SUCCESS
        record.steps_taken = 1

        eval_res = evaluator.evaluate(record)
        assert eval_res.success is False
        assert eval_res.details["verification_run"] is True
        assert eval_res.details["verification_succeeded"] is False
        assert "Final verification command failed" in eval_res.reason

    def test_is_clean_property(self, evaluator):
        clean_res = EvaluationResult(
            success=True,
            score=1.0,
            reason="Clean",
            details={"hit_max_steps": False, "has_error": False, "failed_tool_count": 0},
        )
        assert clean_res.is_clean is True

        dirty_res = EvaluationResult(
            success=True,
            score=0.7,
            reason="1 tool error",
            details={"hit_max_steps": False, "has_error": False, "failed_tool_count": 1},
        )
        assert dirty_res.is_clean is False

    def test_batch_evaluate(self, evaluator):
        r1 = RunRecord(task="t1")
        r1.outcome = RunOutcome.SUCCESS
        r2 = RunRecord(task="t2")
        r2.outcome = RunOutcome.MAX_STEPS_REACHED

        results = evaluator.batch_evaluate([r1, r2])
        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False

