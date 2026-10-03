"""Tests for TrajectoryStore module."""

import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from coding_agent.models import AgentResult
from coding_agent.run_record import RunEvent, RunEventType, RunOutcome, RunRecord
from coding_agent.trajectory_store import TrajectoryStore


@pytest.fixture
def temp_store():
    """Fixture providing a TrajectoryStore backed by a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = TrajectoryStore(storage_dir=tmp_dir)
        yield store


@pytest.fixture
def sample_record():
    """Fixture providing a populated RunRecord for testing."""
    record = RunRecord(task="Test trajectory recording task")
    record.record_task_start("Test trajectory recording task")
    record.record_llm_call(step=1, message_count=1, has_tools=True)
    record.record_tool_call(
        step=1,
        tool_name="read_file",
        arguments={"path": "test.txt"},
        tool_call_id="call_1",
        approved=None,
    )
    record.record_tool_result(
        step=1,
        tool_name="read_file",
        tool_call_id="call_1",
        output="file content",
        is_error=False,
    )
    record.record_final_response("Task completed successfully.")
    record.outcome = RunOutcome.SUCCESS
    record.steps_taken = 1
    record.metadata = {"model": "test-model"}
    return record


class TestTrajectoryStore:
    def test_default_storage_dir(self):
        store = TrajectoryStore()
        expected = Path.home() / ".coding_agent" / "trajectories"
        assert store.storage_dir == expected

    def test_custom_storage_dir(self, temp_store):
        assert temp_store.storage_dir.is_dir() or not temp_store.storage_dir.exists()

    def test_get_path(self, temp_store):
        path = temp_store.get_path("run-123")
        assert path == temp_store.storage_dir / "run-123.json"

    def test_save_creates_directory_and_writes_json(self, temp_store, sample_record):
        filepath = temp_store.save(sample_record)
        assert filepath.is_file()
        assert filepath == temp_store.get_path(sample_record.run_id)

        # Verify contents are valid JSON
        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert data["run_id"] == sample_record.run_id
        assert data["task"] == "Test trajectory recording task"
        assert data["outcome"] == "success"
        assert len(data["events"]) == 5

    def test_load_round_trip(self, temp_store, sample_record):
        temp_store.save(sample_record)
        loaded = temp_store.load(sample_record.run_id)

        assert loaded.run_id == sample_record.run_id
        assert loaded.task == sample_record.task
        assert loaded.outcome == sample_record.outcome
        assert loaded.final_response == sample_record.final_response
        assert loaded.steps_taken == sample_record.steps_taken
        assert len(loaded.events) == len(sample_record.events)
        assert loaded.metadata == sample_record.metadata

    def test_load_nonexistent_raises_file_not_found(self, temp_store):
        with pytest.raises(FileNotFoundError, match="No trajectory found"):
            temp_store.load("nonexistent-id")

    def test_exists(self, temp_store, sample_record):
        assert not temp_store.exists(sample_record.run_id)
        temp_store.save(sample_record)
        assert temp_store.exists(sample_record.run_id)

    def test_list_run_ids(self, temp_store):
        assert temp_store.list_run_ids() == []

        rec1 = RunRecord(run_id="run-aaa")
        rec2 = RunRecord(run_id="run-bbb")
        temp_store.save(rec1)
        temp_store.save(rec2)

        ids = temp_store.list_run_ids()
        assert ids == ["run-aaa", "run-bbb"]

    def test_list_records(self, temp_store):
        rec1 = RunRecord(run_id="run-1", task="Task 1")
        rec2 = RunRecord(run_id="run-2", task="Task 2")
        temp_store.save(rec1)
        temp_store.save(rec2)

        records = temp_store.list_records()
        assert len(records) == 2
        tasks = [r.task for r in records]
        assert "Task 1" in tasks
        assert "Task 2" in tasks

    def test_delete(self, temp_store, sample_record):
        assert not temp_store.delete(sample_record.run_id)
        temp_store.save(sample_record)
        assert temp_store.exists(sample_record.run_id)
        assert temp_store.delete(sample_record.run_id)
        assert not temp_store.exists(sample_record.run_id)

    def test_overwrite_on_save(self, temp_store, sample_record):
        temp_store.save(sample_record)
        sample_record.final_response = "Updated final response"
        temp_store.save(sample_record)

        loaded = temp_store.load(sample_record.run_id)
        assert loaded.final_response == "Updated final response"


class TestCLITrajectoryIntegration:
    def test_cli_saves_trajectory(self, sample_record):
        from coding_agent.cli import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_agent = MagicMock()
            mock_agent.run.return_value = AgentResult(
                messages=[{"role": "user", "content": "hello"}],
                final_response="Hi there!",
                steps_taken=1,
                completed=True,
                run_record=sample_record,
            )

            with patch("coding_agent.cli.Agent", return_value=mock_agent):
                with patch("coding_agent.cli.load_config_from_env"):
                    exit_code = main(["hello", "--trajectory-dir", tmp_dir])
                    assert exit_code == 0

            store = TrajectoryStore(storage_dir=tmp_dir)
            assert store.exists(sample_record.run_id)
            loaded = store.load(sample_record.run_id)
            assert loaded.task == sample_record.task
