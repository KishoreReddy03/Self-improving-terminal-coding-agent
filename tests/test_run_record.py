"""Focused tests for the RunRecord data model and agent loop instrumentation.

These tests verify:
- RunRecord / RunEvent construction and serialisation
- RunRecord property helpers (tool_calls, error_events, duration, etc.)
- Agent loop populates RunRecord correctly for every execution path:
    success, max_steps, tool approved, tool denied, plan mode, unknown tool,
    tool exception, error outcome.
"""

import json
import unittest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from coding_agent.agent import Agent
from coding_agent.client import LLMClient
from coding_agent.models import AgentConfig, AgentResult, ModelResponse, ToolCall
from coding_agent.registry import ToolRegistry
from coding_agent.run_record import RunEvent, RunEventType, RunOutcome, RunRecord
from coding_agent.tools import Tool


# ---------------------------------------------------------------------------
# Shared test helpers (mirrors the helpers in test_agent.py)
# ---------------------------------------------------------------------------


class FakeTool(Tool):
    """Minimal fake tool for loop instrumentation tests."""

    def __init__(self, name: str, return_value: Any = "ok", is_read_only: bool = True) -> None:
        super().__init__(
            name=name,
            description=f"Fake tool {name}",
            parameters={"type": "object", "properties": {}},
            is_read_only=is_read_only,
        )
        self.return_value = return_value

    def execute(self, **kwargs: Any) -> Any:
        if isinstance(self.return_value, Exception):
            raise self.return_value
        return self.return_value


class FakeLLMClient(LLMClient):
    """Fake client returning a scripted sequence of ModelResponse objects."""

    def __init__(self, responses: List[ModelResponse]) -> None:
        super().__init__(config=AgentConfig())
        self.responses = list(responses)
        self._idx = 0

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        if self._idx < len(self.responses):
            resp = self.responses[self._idx]
            self._idx += 1
            return resp
        return ModelResponse(content="fallback")


# ---------------------------------------------------------------------------
# 1. Unit tests for RunRecord / RunEvent data structures
# ---------------------------------------------------------------------------


class TestRunEvent(unittest.TestCase):
    """Unit tests for the RunEvent dataclass."""

    def test_default_timestamp_is_utc(self):
        event = RunEvent(event_type=RunEventType.TASK_START, step=0)
        self.assertIsNotNone(event.timestamp.tzinfo)

    def test_to_dict_round_trip(self):
        event = RunEvent(
            event_type=RunEventType.TOOL_CALL,
            step=2,
            data={"tool_name": "read_file", "approved": True},
        )
        d = event.to_dict()
        restored = RunEvent.from_dict(d)
        self.assertEqual(restored.event_type, RunEventType.TOOL_CALL)
        self.assertEqual(restored.step, 2)
        self.assertEqual(restored.data["tool_name"], "read_file")
        self.assertEqual(restored.data["approved"], True)

    def test_all_event_type_values_round_trip(self):
        for et in RunEventType:
            event = RunEvent(event_type=et, step=0, data={})
            restored = RunEvent.from_dict(event.to_dict())
            self.assertEqual(restored.event_type, et)

    def test_invalid_event_type_raises(self):
        with self.assertRaises(ValueError):
            RunEvent.from_dict({"event_type": "NONEXISTENT", "step": 0, "data": {}, "timestamp": datetime.now(timezone.utc).isoformat()})


class TestRunRecord(unittest.TestCase):
    """Unit tests for the RunRecord dataclass and its helpers."""

    def _make_complete_record(self) -> RunRecord:
        """Build a fully-populated RunRecord for serialisation tests."""
        record = RunRecord()
        record.record_task_start("fix the bug")
        record.record_llm_call(step=1, message_count=2, has_tools=True)
        record.record_tool_call(
            step=1,
            tool_name="read_file",
            arguments={"path": "foo.py"},
            tool_call_id="c1",
            approved=None,
        )
        record.record_tool_result(
            step=1,
            tool_name="read_file",
            tool_call_id="c1",
            output="def foo(): pass",
            is_error=False,
        )
        record.record_llm_call(step=2, message_count=4, has_tools=True)
        record.record_tool_call(
            step=2,
            tool_name="write_file",
            arguments={"path": "bar.py", "content": "x"},
            tool_call_id="c2",
            approved=True,
        )
        record.record_tool_result(
            step=2,
            tool_name="write_file",
            tool_call_id="c2",
            output="Written.",
            is_error=False,
        )
        record.record_final_response("All done.")
        record.outcome = RunOutcome.SUCCESS
        record.final_response = "All done."
        record.steps_taken = 2
        record.finished_at = datetime.now(timezone.utc)
        return record

    # -- basic construction ---------------------------------------------------

    def test_run_id_is_auto_generated(self):
        r1 = RunRecord()
        r2 = RunRecord()
        self.assertNotEqual(r1.run_id, r2.run_id)

    def test_task_start_sets_task_and_appends_event(self):
        record = RunRecord()
        record.record_task_start("write tests")
        self.assertEqual(record.task, "write tests")
        self.assertEqual(len(record.events), 1)
        self.assertEqual(record.events[0].event_type, RunEventType.TASK_START)
        self.assertEqual(record.events[0].data["task"], "write tests")

    def test_llm_call_event_appended(self):
        record = RunRecord()
        record.record_llm_call(step=1, message_count=3, has_tools=False)
        self.assertEqual(len(record.events), 1)
        ev = record.events[0]
        self.assertEqual(ev.event_type, RunEventType.LLM_CALL)
        self.assertEqual(ev.step, 1)
        self.assertEqual(ev.data["message_count"], 3)
        self.assertFalse(ev.data["has_tools"])

    def test_tool_call_event_captures_approval(self):
        record = RunRecord()
        record.record_tool_call(
            step=1, tool_name="write_file", arguments={"path": "a.py"},
            tool_call_id="x1", approved=False,
        )
        ev = record.events[0]
        self.assertEqual(ev.event_type, RunEventType.TOOL_CALL)
        self.assertFalse(ev.data["approved"])
        self.assertEqual(ev.data["tool_name"], "write_file")

    def test_tool_result_is_error_flag(self):
        record = RunRecord()
        record.record_tool_result(
            step=1, tool_name="write_file", tool_call_id="x1",
            output="Error: denied.", is_error=True,
        )
        ev = record.events[0]
        self.assertEqual(ev.event_type, RunEventType.TOOL_RESULT)
        self.assertTrue(ev.data["is_error"])

    def test_final_response_event(self):
        record = RunRecord()
        record.record_final_response("The answer is 42.")
        ev = record.events[0]
        self.assertEqual(ev.event_type, RunEventType.FINAL_RESPONSE)
        self.assertEqual(ev.data["content"], "The answer is 42.")

    def test_error_event(self):
        record = RunRecord()
        record.record_error("ValueError", "bad input")
        ev = record.events[0]
        self.assertEqual(ev.event_type, RunEventType.ERROR_EVENT)
        self.assertEqual(ev.data["error_type"], "ValueError")
        self.assertEqual(ev.data["message"], "bad input")

    # -- property helpers -----------------------------------------------------

    def test_tool_calls_property_filters_correctly(self):
        record = self._make_complete_record()
        calls = record.tool_calls
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(e.event_type == RunEventType.TOOL_CALL for e in calls))

    def test_tool_results_property_filters_correctly(self):
        record = self._make_complete_record()
        results = record.tool_results
        self.assertEqual(len(results), 2)
        self.assertTrue(all(e.event_type == RunEventType.TOOL_RESULT for e in results))

    def test_llm_calls_property(self):
        record = self._make_complete_record()
        self.assertEqual(len(record.llm_calls), 2)

    def test_was_successful_true_for_success_outcome(self):
        record = RunRecord()
        record.outcome = RunOutcome.SUCCESS
        self.assertTrue(record.was_successful())

    def test_was_successful_false_for_other_outcomes(self):
        for outcome in (RunOutcome.MAX_STEPS_REACHED, RunOutcome.ERROR, None):
            record = RunRecord()
            record.outcome = outcome
            self.assertFalse(record.was_successful())

    def test_tool_call_count(self):
        record = self._make_complete_record()
        self.assertEqual(record.tool_call_count(), 2)

    def test_error_tool_results_filters_errors(self):
        record = RunRecord()
        record.record_tool_result(step=1, tool_name="t1", tool_call_id="c1", output="ok", is_error=False)
        record.record_tool_result(step=1, tool_name="t2", tool_call_id="c2", output="Error: x", is_error=True)
        errs = record.error_tool_results()
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0].data["tool_name"], "t2")

    def test_duration_seconds_none_while_unfinished(self):
        record = RunRecord()
        self.assertIsNone(record.duration_seconds)

    def test_duration_seconds_computed_correctly(self):
        record = RunRecord()
        record.finished_at = datetime.now(timezone.utc)
        dur = record.duration_seconds
        self.assertIsNotNone(dur)
        self.assertGreaterEqual(dur, 0.0)

    # -- serialisation --------------------------------------------------------

    def test_to_dict_and_from_dict_round_trip(self):
        record = self._make_complete_record()
        d = record.to_dict()
        restored = RunRecord.from_dict(d)

        self.assertEqual(restored.run_id, record.run_id)
        self.assertEqual(restored.task, record.task)
        self.assertEqual(restored.outcome, record.outcome)
        self.assertEqual(restored.final_response, record.final_response)
        self.assertEqual(restored.steps_taken, record.steps_taken)
        self.assertEqual(len(restored.events), len(record.events))

    def test_to_json_produces_valid_json(self):
        record = self._make_complete_record()
        raw = record.to_json()
        data = json.loads(raw)
        self.assertEqual(data["outcome"], "success")
        self.assertEqual(data["task"], "fix the bug")

    def test_from_json_round_trip(self):
        record = self._make_complete_record()
        restored = RunRecord.from_json(record.to_json())
        self.assertEqual(restored.run_id, record.run_id)
        self.assertEqual(restored.outcome, RunOutcome.SUCCESS)
        self.assertEqual(len(restored.events), len(record.events))

    def test_outcome_none_serialises_as_null(self):
        record = RunRecord()
        d = record.to_dict()
        self.assertIsNone(d["outcome"])
        restored = RunRecord.from_dict(d)
        self.assertIsNone(restored.outcome)

    def test_metadata_round_trips(self):
        record = RunRecord(metadata={"model": "gpt-4o", "temperature": 0.5})
        restored = RunRecord.from_dict(record.to_dict())
        self.assertEqual(restored.metadata["model"], "gpt-4o")
        self.assertAlmostEqual(restored.metadata["temperature"], 0.5)


# ---------------------------------------------------------------------------
# 2. Integration tests — agent loop populates RunRecord correctly
# ---------------------------------------------------------------------------


def _make_agent(responses: List[ModelResponse], registry: Optional[ToolRegistry] = None, **kwargs) -> Agent:
    client = FakeLLMClient(responses)
    return Agent(client=client, registry=registry or ToolRegistry(), **kwargs)


class TestAgentRunRecordIntegration(unittest.TestCase):
    """Verify the agent loop populates RunRecord correctly for every outcome."""

    # -- TASK_START -----------------------------------------------------------

    def test_task_start_event_recorded(self):
        agent = _make_agent([ModelResponse(content="Hello!")])
        result = agent.run("Do something")
        record = result.run_record
        self.assertIsNotNone(record)
        task_events = [e for e in record.events if e.event_type == RunEventType.TASK_START]
        self.assertEqual(len(task_events), 1)
        self.assertEqual(task_events[0].data["task"], "Do something")

    # -- SUCCESS outcome ------------------------------------------------------

    def test_success_outcome_and_final_response(self):
        agent = _make_agent([ModelResponse(content="All done.")])
        result = agent.run("task")
        record = result.run_record
        self.assertEqual(record.outcome, RunOutcome.SUCCESS)
        self.assertTrue(record.was_successful())
        self.assertEqual(record.final_response, "All done.")
        self.assertEqual(record.steps_taken, 1)
        self.assertIsNotNone(record.finished_at)

    def test_final_response_event_recorded_on_success(self):
        agent = _make_agent([ModelResponse(content="Done.")])
        result = agent.run("task")
        record = result.run_record
        fr_events = [e for e in record.events if e.event_type == RunEventType.FINAL_RESPONSE]
        self.assertEqual(len(fr_events), 1)
        self.assertEqual(fr_events[0].data["content"], "Done.")

    # -- LLM_CALL events ------------------------------------------------------

    def test_llm_call_event_count_matches_steps(self):
        tool = FakeTool(name="t", return_value="r")
        reg = ToolRegistry()
        reg.register(tool)
        responses = [
            ModelResponse(content=None, tool_calls=[ToolCall(name="t", arguments={}, id="c1")]),
            ModelResponse(content="Finished."),
        ]
        agent = _make_agent(responses, registry=reg)
        result = agent.run("two steps")
        record = result.run_record
        self.assertEqual(len(record.llm_calls), 2)
        self.assertEqual(record.llm_calls[0].step, 1)
        self.assertEqual(record.llm_calls[1].step, 2)

    # -- TOOL_CALL / TOOL_RESULT for read-only tool ---------------------------

    def test_read_only_tool_approved_none_in_record(self):
        tool = FakeTool(name="read_file", return_value="content", is_read_only=True)
        reg = ToolRegistry()
        reg.register(tool)
        responses = [
            ModelResponse(content=None, tool_calls=[ToolCall(name="read_file", arguments={"path": "x"}, id="c1")]),
            ModelResponse(content="Done."),
        ]
        agent = _make_agent(responses, registry=reg)
        result = agent.run("read")
        record = result.run_record
        tc_events = record.tool_calls
        self.assertEqual(len(tc_events), 1)
        # Read-only tools skip the approval gate — approved should be None
        self.assertIsNone(tc_events[0].data["approved"])
        # Corresponding result should not be an error
        tr_events = record.tool_results
        self.assertEqual(len(tr_events), 1)
        self.assertFalse(tr_events[0].data["is_error"])
        self.assertEqual(tr_events[0].data["output"], "content")

    # -- TOOL_CALL / TOOL_RESULT for write tool — approved --------------------

    def test_approved_write_tool_recorded(self):
        tool = FakeTool(name="write_file", return_value="Written.", is_read_only=False)
        reg = ToolRegistry()
        reg.register(tool)
        responses = [
            ModelResponse(content=None, tool_calls=[ToolCall(name="write_file", arguments={}, id="w1")]),
            ModelResponse(content="Done."),
        ]
        agent = _make_agent(responses, registry=reg, approval_callback=lambda n, k: True)
        result = agent.run("write")
        record = result.run_record
        tc = record.tool_calls[0]
        tr = record.tool_results[0]
        self.assertTrue(tc.data["approved"])
        self.assertFalse(tr.data["is_error"])
        self.assertEqual(tr.data["output"], "Written.")

    # -- TOOL_CALL / TOOL_RESULT for write tool — denied ----------------------

    def test_denied_write_tool_recorded_as_error(self):
        tool = FakeTool(name="write_file", return_value="Written.", is_read_only=False)
        reg = ToolRegistry()
        reg.register(tool)
        responses = [
            ModelResponse(content=None, tool_calls=[ToolCall(name="write_file", arguments={}, id="w2")]),
            ModelResponse(content="Handled denial."),
        ]
        agent = _make_agent(responses, registry=reg, approval_callback=lambda n, k: False)
        result = agent.run("write")
        record = result.run_record
        tc = record.tool_calls[0]
        tr = record.tool_results[0]
        self.assertFalse(tc.data["approved"])
        self.assertTrue(tr.data["is_error"])
        self.assertIn("denied", tr.data["output"])

    # -- TOOL_CALL in plan mode -----------------------------------------------

    def test_plan_mode_write_blocked_approved_false_in_record(self):
        tool = FakeTool(name="write_file", return_value="Written.", is_read_only=False)
        reg = ToolRegistry()
        reg.register(tool)
        responses = [
            ModelResponse(content=None, tool_calls=[ToolCall(name="write_file", arguments={}, id="pm1")]),
            ModelResponse(content="Acknowledged."),
        ]
        agent = _make_agent(responses, registry=reg, plan_mode=True)
        result = agent.run("write in plan mode")
        record = result.run_record
        tc = record.tool_calls[0]
        tr = record.tool_results[0]
        self.assertFalse(tc.data["approved"])
        self.assertTrue(tr.data["is_error"])
        self.assertIn("disabled in plan mode", tr.data["output"])

    # -- Unknown tool ---------------------------------------------------------

    def test_unknown_tool_recorded_with_error_result(self):
        responses = [
            ModelResponse(content=None, tool_calls=[ToolCall(name="ghost_tool", arguments={}, id="g1")]),
            ModelResponse(content="Recovered."),
        ]
        agent = _make_agent(responses)
        result = agent.run("call unknown")
        record = result.run_record
        tc = record.tool_calls[0]
        tr = record.tool_results[0]
        self.assertEqual(tc.data["tool_name"], "ghost_tool")
        self.assertIsNone(tc.data["approved"])  # no approval gate for missing tools
        self.assertTrue(tr.data["is_error"])
        self.assertIn("not found", tr.data["output"])

    # -- Tool exception -------------------------------------------------------

    def test_tool_exception_recorded_as_error_result(self):
        boom = FakeTool(name="boom", return_value=RuntimeError("kaboom"), is_read_only=True)
        reg = ToolRegistry()
        reg.register(boom)
        responses = [
            ModelResponse(content=None, tool_calls=[ToolCall(name="boom", arguments={}, id="b1")]),
            ModelResponse(content="Recovered."),
        ]
        agent = _make_agent(responses, registry=reg)
        result = agent.run("boom")
        record = result.run_record
        tr = record.tool_results[0]
        self.assertTrue(tr.data["is_error"])
        self.assertIn("kaboom", tr.data["output"])

    # -- MAX_STEPS outcome ----------------------------------------------------

    def test_max_steps_outcome_and_event_count(self):
        repeating = ModelResponse(
            content=None,
            tool_calls=[ToolCall(name="t", arguments={}, id="loop")],
        )
        reg = ToolRegistry()
        reg.register(FakeTool(name="t"))
        agent = _make_agent([repeating] * 10, registry=reg, config=AgentConfig(max_steps=3))
        result = agent.run("loop")
        record = result.run_record
        self.assertEqual(record.outcome, RunOutcome.MAX_STEPS_REACHED)
        self.assertEqual(record.steps_taken, 3)
        self.assertFalse(record.was_successful())
        self.assertIsNotNone(record.finished_at)

    # -- run_record attached to AgentResult -----------------------------------

    def test_agent_result_carries_run_record(self):
        agent = _make_agent([ModelResponse(content="Hi!")])
        result = agent.run("hello")
        self.assertIsNotNone(result.run_record)
        self.assertIsInstance(result.run_record, RunRecord)

    # -- messages snapshot on record ------------------------------------------

    def test_run_record_messages_match_agent_result_messages(self):
        agent = _make_agent([ModelResponse(content="Hi!")])
        result = agent.run("hello")
        record = result.run_record
        self.assertEqual(len(record.messages), len(result.messages))

    # -- task derived from conversation list ----------------------------------

    def test_task_derived_from_conversation_list(self):
        conversation = [{"role": "user", "content": "describe the architecture"}]
        agent = _make_agent([ModelResponse(content="Here it is.")])
        result = agent.run(conversation)
        record = result.run_record
        self.assertEqual(record.task, "describe the architecture")

    # -- auto-approve writes --------------------------------------------------

    def test_auto_approve_write_has_approved_true_in_record(self):
        tool = FakeTool(name="write_file", return_value="Written.", is_read_only=False)
        reg = ToolRegistry()
        reg.register(tool)
        responses = [
            ModelResponse(content=None, tool_calls=[ToolCall(name="write_file", arguments={}, id="a1")]),
            ModelResponse(content="Done."),
        ]
        agent = _make_agent(responses, registry=reg, auto_approve=True)
        result = agent.run("write auto")
        record = result.run_record
        tc = record.tool_calls[0]
        # Auto-approved writes should record approved=True
        self.assertTrue(tc.data["approved"])

    # -- serialisation of agent-produced record --------------------------------

    def test_agent_run_record_round_trips_to_json(self):
        tool = FakeTool(name="read_file", return_value="data", is_read_only=True)
        reg = ToolRegistry()
        reg.register(tool)
        responses = [
            ModelResponse(content=None, tool_calls=[ToolCall(name="read_file", arguments={"path": "f.py"}, id="r1")]),
            ModelResponse(content="Here you go."),
        ]
        agent = _make_agent(responses, registry=reg)
        result = agent.run("read it")
        record = result.run_record

        # Must be fully serialisable
        raw_json = record.to_json()
        data = json.loads(raw_json)
        self.assertEqual(data["outcome"], "success")

        # Must round-trip
        restored = RunRecord.from_json(raw_json)
        self.assertEqual(restored.run_id, record.run_id)
        self.assertEqual(len(restored.events), len(record.events))
        self.assertEqual(restored.tool_call_count(), 1)


if __name__ == "__main__":
    unittest.main()
