"""Unit tests for Experience and ExperienceMemory."""

import json
import tempfile

pytest_plugins = []

from coding_agent.evaluator import EvaluationResult
from coding_agent.memory import Experience, ExperienceMemory
from coding_agent.reflection import ReflectionResult


def make_sample_experience(
    task: str = "Fix bug in parser module",
    success: bool = True,
    score: float = 1.0,
    what_worked: str = "Used read_file to inspect parser.py then edited line 42.",
    what_failed: str = "None.",
    why_it_failed: str = "N/A",
    what_to_do_differently: str = "Always inspect file first.",
) -> Experience:
    evaluation = EvaluationResult(
        success=success,
        score=score,
        reason="Run completed cleanly." if success else "Run hit max steps.",
        details={"completed": success, "failed_tool_count": 0 if success else 1},
    )
    reflection = ReflectionResult(
        what_worked=what_worked,
        what_failed=what_failed,
        why_it_failed=why_it_failed,
        what_to_do_differently=what_to_do_differently,
        summary="Experience summary.",
    )
    return Experience(
        task=task,
        evaluation=evaluation,
        reflection=reflection,
        trajectory_run_id="run-uuid-1234",
    )


class TestExperienceSerialization:
    def test_to_dict_and_from_dict(self):
        exp = make_sample_experience()
        d = exp.to_dict()
        assert d["task"] == "Fix bug in parser module"
        assert d["evaluation"]["success"] is True
        assert d["reflection"]["what_worked"] == "Used read_file to inspect parser.py then edited line 42."

        reconstructed = Experience.from_dict(d)
        assert reconstructed.experience_id == exp.experience_id
        assert reconstructed.task == exp.task
        assert reconstructed.evaluation == exp.evaluation
        assert reconstructed.reflection == exp.reflection

    def test_to_json_and_from_json(self):
        exp = make_sample_experience(task="Refactor database client")
        json_str = exp.to_json(indent=2)
        parsed = json.loads(json_str)
        assert parsed["task"] == "Refactor database client"

        reconstructed = Experience.from_json(json_str)
        assert reconstructed.experience_id == exp.experience_id
        assert reconstructed.task == exp.task


class TestExperienceMemory:
    def test_save_and_get_experience(self):
        exp = make_sample_experience(task="Build CLI interface")
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory = ExperienceMemory(storage_dir=tmp_dir)
            filepath = memory.save(exp)

            assert filepath.is_file()
            assert memory.exists(exp.experience_id)

            retrieved = memory.get(exp.experience_id)
            assert retrieved.experience_id == exp.experience_id
            assert retrieved.task == exp.task
            assert retrieved.evaluation.score == 1.0

    def test_add_convenience_method(self):
        eval_res = EvaluationResult(success=True, score=0.9, reason="Good", details={})
        refl_res = ReflectionResult(
            what_worked="Fast execution",
            what_failed="None",
            why_it_failed="N/A",
            what_to_do_differently="None",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory = ExperienceMemory(storage_dir=tmp_dir)
            exp = memory.add(
                task="Write unit tests for math helper",
                evaluation=eval_res,
                reflection=refl_res,
                trajectory_run_id="run-5678",
            )
            assert memory.exists(exp.experience_id)
            retrieved = memory.get(exp.experience_id)
            assert retrieved.task == "Write unit tests for math helper"

    def test_list_experiences(self):
        exp1 = make_sample_experience(task="Task 1")
        exp2 = make_sample_experience(task="Task 2")
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory = ExperienceMemory(storage_dir=tmp_dir)
            memory.save(exp1)
            memory.save(exp2)

            experiences = memory.list_experiences()
            assert len(experiences) == 2
            tasks = [e.task for e in experiences]
            assert "Task 1" in tasks
            assert "Task 2" in tasks

    def test_delete_experience(self):
        exp = make_sample_experience()
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory = ExperienceMemory(storage_dir=tmp_dir)
            memory.save(exp)
            assert memory.exists(exp.experience_id)

            assert memory.delete(exp.experience_id) is True
            assert memory.exists(exp.experience_id) is False
            assert memory.delete(exp.experience_id) is False

    def test_retrieve_relevant_experiences(self):
        exp_parser = make_sample_experience(
            task="Fix JSON parser bug in response parser",
            what_worked="Inspected JSON string and handled double quotes.",
        )
        exp_shell = make_sample_experience(
            task="Execute bash shell script for build system",
            what_worked="Ran subprocess with timeout parameter.",
        )
        exp_db = make_sample_experience(
            task="Setup SQLite database connection and migrations",
            what_worked="Created tables and indexes.",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            memory = ExperienceMemory(storage_dir=tmp_dir)
            memory.save(exp_parser)
            memory.save(exp_shell)
            memory.save(exp_db)

            # Query relevant to JSON parser
            results = memory.retrieve_relevant("Debug JSON parsing issue in parser", limit=2)
            assert len(results) >= 1
            assert results[0].experience_id == exp_parser.experience_id

            # Query relevant to shell script
            results_shell = memory.retrieve_relevant("Run shell command script", limit=1)
            assert len(results_shell) == 1
            assert results_shell[0].experience_id == exp_shell.experience_id

    def test_retrieve_relevant_filtering_options(self):
        exp_failed = make_sample_experience(
            task="Broken database migration",
            success=False,
            score=0.2,
        )
        exp_success = make_sample_experience(
            task="Successful database query optimization",
            success=True,
            score=1.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            memory = ExperienceMemory(storage_dir=tmp_dir)
            memory.save(exp_failed)
            memory.save(exp_success)

            # Filter only successful
            only_succ = memory.retrieve_relevant("database", only_successful=True)
            assert len(only_succ) == 1
            assert only_succ[0].experience_id == exp_success.experience_id

            # Filter by min_score
            high_score = memory.retrieve_relevant("database", min_score=0.8)
            assert len(high_score) == 1
            assert high_score[0].experience_id == exp_success.experience_id
