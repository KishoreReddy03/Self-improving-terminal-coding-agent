"""Tests for ExperiencePipeline and PipelineResult."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from coding_agent.evaluator import EvaluationResult, TrajectoryEvaluator
from coding_agent.memory import ExperienceMemory
from coding_agent.models import AgentResult
from coding_agent.pipeline import ExperiencePipeline, PipelineResult
from coding_agent.reflection import ReflectionGenerator, ReflectionResult
from coding_agent.run_record import RunOutcome, RunRecord
from coding_agent.trajectory_store import TrajectoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_successful_record(task: str = "Write hello world") -> RunRecord:
    """Return a RunRecord that looks like a clean successful run."""
    record = RunRecord()
    record.record_task_start(task)
    record.record_llm_call(step=1, message_count=1, has_tools=False)
    record.record_final_response("Hello, world!")
    record.final_response = "Hello, world!"
    record.steps_taken = 1
    record.outcome = RunOutcome.SUCCESS
    return record


def _make_failed_record(task: str = "Do something") -> RunRecord:
    """Return a RunRecord that looks like a failed run."""
    record = RunRecord()
    record.record_task_start(task)
    record.record_llm_call(step=1, message_count=1, has_tools=True)
    record.record_tool_call(
        step=1,
        tool_name="shell",
        arguments={"command": "rm -rf /"},
        tool_call_id="c1",
        approved=True,
    )
    record.record_tool_result(
        step=1,
        tool_name="shell",
        tool_call_id="c1",
        output="Error: Permission denied.",
        is_error=True,
    )
    record.outcome = RunOutcome.ERROR
    record.error_message = "Permission denied."
    record.steps_taken = 1
    return record


def _make_agent_result(record: RunRecord, completed: bool = True) -> AgentResult:
    return AgentResult(
        final_response=record.final_response,
        messages=[],
        steps_taken=record.steps_taken,
        completed=completed,
        run_record=record,
    )


def _fake_reflection() -> ReflectionResult:
    return ReflectionResult(
        what_worked="Executed cleanly.",
        what_failed="Nothing.",
        why_it_failed="N/A",
        what_to_do_differently="Keep it up.",
        summary="Good run.",
    )


def _mock_reflection_generator() -> ReflectionGenerator:
    gen = MagicMock(spec=ReflectionGenerator)
    gen.generate.return_value = _fake_reflection()
    return gen


# ---------------------------------------------------------------------------
# PipelineResult tests
# ---------------------------------------------------------------------------


class TestPipelineResult:
    def _make_result(self, errors=None) -> PipelineResult:
        record = _make_successful_record()
        evaluation = EvaluationResult(success=True, score=1.0, reason="All good.")
        reflection = _fake_reflection()
        return PipelineResult(
            run_record=record,
            evaluation=evaluation,
            reflection=reflection,
            errors=errors or [],
        )

    def test_success_property_true_when_no_errors(self):
        result = self._make_result(errors=[])
        assert result.success is True

    def test_success_property_false_when_errors(self):
        result = self._make_result(errors=["Something failed"])
        assert result.success is False

    def test_to_dict_contains_expected_keys(self):
        result = self._make_result()
        d = result.to_dict()
        assert "run_id" in d
        assert "evaluation" in d
        assert "reflection" in d
        assert "experience_id" in d
        assert "trajectory_path" in d
        assert "experience_path" in d
        assert "errors" in d

    def test_to_dict_experience_id_none_when_no_experience(self):
        result = self._make_result()
        assert result.experience is None
        assert result.to_dict()["experience_id"] is None

    def test_to_dict_trajectory_path_none_by_default(self):
        result = self._make_result()
        assert result.trajectory_path is None
        assert result.to_dict()["trajectory_path"] is None


# ---------------------------------------------------------------------------
# ExperiencePipeline – core process() tests
# ---------------------------------------------------------------------------


class TestExperiencePipelineProcess:
    def test_raises_if_run_record_is_none(self):
        pipeline = ExperiencePipeline(enable_reflection=False)
        result = AgentResult(
            final_response=None, messages=[], steps_taken=0, completed=False, run_record=None
        )
        with pytest.raises(ValueError, match="run_record is None"):
            pipeline.process(result)

    def test_returns_pipeline_result(self):
        record = _make_successful_record()
        agent_result = _make_agent_result(record)
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            enable_reflection=True,
        )
        pr = pipeline.process(agent_result)
        assert isinstance(pr, PipelineResult)

    def test_evaluation_is_populated(self):
        record = _make_successful_record()
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
        )
        pr = pipeline.process(_make_agent_result(record))
        assert isinstance(pr.evaluation, EvaluationResult)
        assert isinstance(pr.evaluation.score, float)

    def test_reflection_is_populated_when_enabled(self):
        record = _make_successful_record()
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            enable_reflection=True,
        )
        pr = pipeline.process(_make_agent_result(record))
        assert isinstance(pr.reflection, ReflectionResult)
        assert pr.reflection.what_worked == "Executed cleanly."

    def test_reflection_placeholder_when_disabled(self):
        record = _make_successful_record()
        pipeline = ExperiencePipeline(enable_reflection=False)
        pr = pipeline.process(_make_agent_result(record))
        assert "disabled" in pr.reflection.summary.lower()
        assert pr.reflection.what_failed == "N/A"

    def test_reflection_error_captured_in_errors_list(self):
        record = _make_successful_record()
        gen = MagicMock(spec=ReflectionGenerator)
        gen.generate.side_effect = RuntimeError("LLM unavailable")
        pipeline = ExperiencePipeline(
            reflection_generator=gen,
            enable_reflection=True,
        )
        pr = pipeline.process(_make_agent_result(record))
        assert len(pr.errors) == 1
        assert "Reflection generation failed" in pr.errors[0]
        assert pr.reflection is not None  # placeholder still provided

    def test_no_trajectory_path_when_store_is_none(self):
        record = _make_successful_record()
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            trajectory_store=None,
        )
        pr = pipeline.process(_make_agent_result(record))
        assert pr.trajectory_path is None

    def test_no_experience_when_memory_is_none(self):
        record = _make_successful_record()
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            memory=None,
        )
        pr = pipeline.process(_make_agent_result(record))
        assert pr.experience is None
        assert pr.experience_path is None


# ---------------------------------------------------------------------------
# ExperiencePipeline – trajectory persistence
# ---------------------------------------------------------------------------


class TestExperiencePipelineTrajectory:
    def test_trajectory_saved_to_disk(self, tmp_path):
        record = _make_successful_record()
        store = TrajectoryStore(storage_dir=tmp_path / "traj")
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            trajectory_store=store,
        )
        pr = pipeline.process(_make_agent_result(record))
        assert pr.trajectory_path is not None
        assert pr.trajectory_path.is_file()

    def test_trajectory_file_contains_evaluation(self, tmp_path):
        record = _make_successful_record()
        store = TrajectoryStore(storage_dir=tmp_path / "traj")
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            trajectory_store=store,
        )
        pr = pipeline.process(_make_agent_result(record))
        content = json.loads(pr.trajectory_path.read_text())
        assert "evaluation" in content.get("metadata", {})

    def test_trajectory_file_contains_reflection(self, tmp_path):
        record = _make_successful_record()
        store = TrajectoryStore(storage_dir=tmp_path / "traj")
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            trajectory_store=store,
        )
        pr = pipeline.process(_make_agent_result(record))
        content = json.loads(pr.trajectory_path.read_text())
        assert "reflection" in content.get("metadata", {})

    def test_trajectory_error_captured_not_raised(self, tmp_path):
        record = _make_successful_record()
        bad_store = MagicMock(spec=TrajectoryStore)
        bad_store.save.side_effect = OSError("Disk full")
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            trajectory_store=bad_store,
        )
        pr = pipeline.process(_make_agent_result(record))
        assert pr.trajectory_path is None
        assert any("Trajectory persistence failed" in e for e in pr.errors)


# ---------------------------------------------------------------------------
# ExperiencePipeline – experience memory storage
# ---------------------------------------------------------------------------


class TestExperiencePipelineMemory:
    def test_experience_saved_to_memory(self, tmp_path):
        record = _make_successful_record(task="Sort a list of numbers")
        mem = ExperienceMemory(storage_dir=tmp_path / "mem")
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            memory=mem,
        )
        pr = pipeline.process(_make_agent_result(record))
        assert pr.experience is not None
        assert pr.experience_path is not None
        assert pr.experience_path.is_file()

    def test_experience_retrievable_from_memory(self, tmp_path):
        record = _make_successful_record(task="Sort a list of numbers")
        mem = ExperienceMemory(storage_dir=tmp_path / "mem")
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            memory=mem,
        )
        pr = pipeline.process(_make_agent_result(record))
        loaded = mem.get(pr.experience.experience_id)
        assert loaded.task == "Sort a list of numbers"

    def test_experience_links_to_run_id(self, tmp_path):
        record = _make_successful_record()
        mem = ExperienceMemory(storage_dir=tmp_path / "mem")
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            memory=mem,
        )
        pr = pipeline.process(_make_agent_result(record))
        assert pr.experience.trajectory_run_id == record.run_id

    def test_only_store_successful_skips_failed_run(self, tmp_path):
        record = _make_failed_record()
        mem = ExperienceMemory(storage_dir=tmp_path / "mem")
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            memory=mem,
            only_store_successful=True,
        )
        pr = pipeline.process(_make_agent_result(record, completed=False))
        assert pr.experience is None
        assert len(mem.list_experiences()) == 0

    def test_only_store_successful_saves_successful_run(self, tmp_path):
        record = _make_successful_record()
        mem = ExperienceMemory(storage_dir=tmp_path / "mem")
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            memory=mem,
            only_store_successful=True,
        )
        pr = pipeline.process(_make_agent_result(record))
        assert pr.experience is not None

    def test_memory_error_captured_not_raised(self, tmp_path):
        record = _make_successful_record()
        bad_mem = MagicMock(spec=ExperienceMemory)
        bad_mem.add.side_effect = OSError("Disk full")
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            memory=bad_mem,
        )
        pr = pipeline.process(_make_agent_result(record))
        assert pr.experience is None
        assert any("Experience storage failed" in e for e in pr.errors)


# ---------------------------------------------------------------------------
# ExperiencePipeline – process_record() helper
# ---------------------------------------------------------------------------


class TestExperiencePipelineProcessRecord:
    def test_process_record_returns_pipeline_result(self):
        record = _make_successful_record()
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
        )
        pr = pipeline.process_record(record)
        assert isinstance(pr, PipelineResult)

    def test_process_record_task_override(self, tmp_path):
        record = _make_successful_record(task="Original task")
        mem = ExperienceMemory(storage_dir=tmp_path / "mem")
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            memory=mem,
        )
        pr = pipeline.process_record(record, task="Override task")
        assert pr.experience.task == "Override task"


# ---------------------------------------------------------------------------
# ExperiencePipeline – full end-to-end integration
# ---------------------------------------------------------------------------


class TestExperiencePipelineEndToEnd:
    def test_full_pipeline_produces_no_errors(self, tmp_path):
        record = _make_successful_record(task="Parse JSON input and validate schema")
        store = TrajectoryStore(storage_dir=tmp_path / "traj")
        mem = ExperienceMemory(storage_dir=tmp_path / "mem")
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            trajectory_store=store,
            memory=mem,
        )
        pr = pipeline.process(_make_agent_result(record))
        assert pr.success is True
        assert pr.trajectory_path is not None
        assert pr.experience is not None

    def test_multiple_runs_accumulate_in_memory(self, tmp_path):
        mem = ExperienceMemory(storage_dir=tmp_path / "mem")
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            memory=mem,
        )
        for task in ["Sort list", "Parse JSON", "Write file"]:
            record = _make_successful_record(task=task)
            pipeline.process(_make_agent_result(record))

        all_experiences = mem.list_experiences()
        assert len(all_experiences) == 3

    def test_pipeline_result_retrievable_experiences_after_storage(self, tmp_path):
        mem = ExperienceMemory(storage_dir=tmp_path / "mem")
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            memory=mem,
        )
        record = _make_successful_record(task="Sort a list of integers in Python")
        pipeline.process(_make_agent_result(record))

        # Retrieve with a related query
        relevant = mem.retrieve_relevant("Sort integers list", limit=1, min_overlap=0.01)
        assert len(relevant) >= 1
        assert "Sort" in relevant[0].task

    def test_verbose_flag_does_not_crash(self, tmp_path, capsys):
        record = _make_successful_record()
        pipeline = ExperiencePipeline(
            reflection_generator=_mock_reflection_generator(),
            verbose=True,
        )
        pipeline.process(_make_agent_result(record))
        captured = capsys.readouterr()
        assert "[pipeline]" in captured.err
