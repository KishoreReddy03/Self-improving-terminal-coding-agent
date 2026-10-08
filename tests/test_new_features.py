"""Unit tests for newly added convenience methods across store, memory, and pipeline."""

import time
import pytest

from coding_agent.evaluator import EvaluationResult
from coding_agent.memory import ExperienceMemory
from coding_agent.models import AgentResult
from coding_agent.pipeline import ExperiencePipeline, PipelineResult
from coding_agent.reflection import ReflectionResult
from coding_agent.run_record import RunRecord
from coding_agent.trajectory_store import TrajectoryStore


class TestTrajectoryStoreNewHelpers:
    def test_count_and_list_records_since(self, tmp_path):
        store = TrajectoryStore(storage_dir=tmp_path / "traj")
        assert store.count() == 0

        from datetime import datetime, timezone
        t0 = time.time() - 100
        r1 = RunRecord(task="task1")
        r1.started_at = datetime.fromtimestamp(t0, tz=timezone.utc)
        r2 = RunRecord(task="task2")
        r2.started_at = datetime.fromtimestamp(t0 + 200, tz=timezone.utc)



        store.save(r1)
        store.save(r2)

        assert store.count() == 2
        since_records = store.list_records_since(t0 + 100)
        assert len(since_records) == 1
        assert since_records[0].task == "task2"


class TestExperienceMemoryPrune:
    def test_prune_by_score_and_failed(self, tmp_path):
        memory = ExperienceMemory(storage_dir=tmp_path / "mem")

        good_eval = EvaluationResult(success=True, score=0.9, reason="good")
        bad_eval = EvaluationResult(success=False, score=0.2, reason="bad")
        dummy_refl = ReflectionResult(
            what_worked="w", what_failed="f", why_it_failed="y", what_to_do_differently="d", summary="s"
        )

        memory.add(task="good task", evaluation=good_eval, reflection=dummy_refl)
        memory.add(task="bad task", evaluation=bad_eval, reflection=dummy_refl)

        assert len(memory.list_experiences()) == 2

        # Prune only failed
        pruned_count = memory.prune(only_failed=True)
        assert pruned_count == 1
        assert len(memory.list_experiences()) == 1
        assert memory.list_experiences()[0].task == "good task"


class TestPipelineNewHelpers:
    def test_pipeline_result_is_actionable(self):
        rec = RunRecord(task="t")
        ev = EvaluationResult(success=True, score=1.0, reason="ok")
        refl = ReflectionResult(
            what_worked="w", what_failed="f", why_it_failed="y", what_to_do_differently="d", summary="s"
        )
        res_no_exp = PipelineResult(run_record=rec, evaluation=ev, reflection=refl, experience=None)
        assert res_no_exp.is_actionable is False

    def test_batch_process(self, tmp_path):
        pipe = ExperiencePipeline(
            trajectory_store=TrajectoryStore(storage_dir=tmp_path / "t"),
            memory=ExperienceMemory(storage_dir=tmp_path / "m"),
            enable_reflection=False,
        )

        r1 = RunRecord(task="t1")
        r1.record_final_response("resp1")
        r2 = RunRecord(task="t2")
        r2.record_final_response("resp2")

        res1 = AgentResult(final_response="resp1", completed=True, run_record=r1)
        res2 = AgentResult(final_response="resp2", completed=True, run_record=r2)

        results = pipe.batch_process([res1, res2])
        assert len(results) == 2
        assert results[0].run_record.task == "t1"
        assert results[1].run_record.task == "t2"
