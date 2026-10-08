"""Tests for WeaknessAnalyzer — corpus-level trajectory weakness detection."""

from __future__ import annotations

from typing import List, Optional

import pytest

from coding_agent.evaluator import EvaluationResult
from coding_agent.run_record import RunEvent, RunEventType, RunOutcome, RunRecord
from coding_agent.weakness_analyzer import (
    ImprovementProposal,
    Severity,
    TrajectoryEntry,
    WeaknessAnalyzer,
    WeaknessEvidence,
    WeaknessReport,
    WeaknessType,
    _detect_bad_tool_selection,
    _detect_failure_to_recover,
    _detect_inefficient_planning,
    _detect_poor_verification,
    _detect_repeated_failed_approaches,
    _detect_unnecessary_tool_calls,
)


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _base_record(task: str = "Do something") -> RunRecord:
    record = RunRecord()
    record.record_task_start(task)
    record.outcome = RunOutcome.SUCCESS
    record.steps_taken = 1
    return record


def _add_tool_call(
    record: RunRecord,
    tool_name: str,
    arguments: dict,
    step: int = 1,
    call_id: Optional[str] = None,
    approved: bool = True,
) -> str:
    cid = call_id or f"c_{len(record.tool_calls)}"
    record.record_tool_call(
        step=step,
        tool_name=tool_name,
        arguments=arguments,
        tool_call_id=cid,
        approved=approved,
    )
    return cid


def _add_tool_result(
    record: RunRecord,
    tool_name: str,
    call_id: str,
    output: str = "ok",
    is_error: bool = False,
    step: int = 1,
) -> None:
    record.record_tool_result(
        step=step,
        tool_name=tool_name,
        tool_call_id=call_id,
        output=output,
        is_error=is_error,
    )


def _entry(record: RunRecord, evaluation: Optional[EvaluationResult] = None) -> TrajectoryEntry:
    return TrajectoryEntry(record=record, evaluation=evaluation)


def _simple_eval(success: bool = True, score: float = 1.0) -> EvaluationResult:
    return EvaluationResult(success=success, score=score, reason="test")


# ===========================================================================
# WeaknessEvidence & ImprovementProposal serialisation
# ===========================================================================


class TestWeaknessEvidenceSerialization:
    def test_to_dict_contains_required_keys(self):
        ev = WeaknessEvidence(
            run_id="abc",
            task_snippet="Sort a list",
            detail="Too many calls",
            metrics={"count": 15},
        )
        d = ev.to_dict()
        assert d["run_id"] == "abc"
        assert d["task_snippet"] == "Sort a list"
        assert d["detail"] == "Too many calls"
        assert d["metrics"]["count"] == 15


class TestImprovementProposalSerialization:
    def _make_proposal(self) -> ImprovementProposal:
        ev = WeaknessEvidence(run_id="r1", task_snippet="x", detail="y")
        return ImprovementProposal(
            weakness_type=WeaknessType.POOR_VERIFICATION,
            severity=Severity.HIGH,
            observed_problem="No tests run",
            evidence=[ev],
            affected_run_count=3,
            corpus_size=5,
            prevalence=0.6,
            likely_cause="Agent doesn't verify",
            proposed_improvement="Run pytest",
            expected_benefit="Fewer regressions",
            risk="Test may not exist",
        )

    def test_to_dict_keys(self):
        p = self._make_proposal()
        d = p.to_dict()
        for key in (
            "weakness_type", "severity", "observed_problem", "evidence",
            "affected_run_count", "corpus_size", "prevalence",
            "likely_cause", "proposed_improvement", "expected_benefit", "risk",
        ):
            assert key in d

    def test_to_json_is_valid_json(self):
        import json
        p = self._make_proposal()
        parsed = json.loads(p.to_json())
        assert parsed["weakness_type"] == "poor_verification"

    def test_prevalence_rounded(self):
        p = self._make_proposal()
        assert p.to_dict()["prevalence"] == round(0.6, 3)


# ===========================================================================
# WeaknessReport serialisation
# ===========================================================================


class TestWeaknessReport:
    def test_to_dict_structure(self):
        report = WeaknessReport(
            proposals=[],
            corpus_size=10,
            analysed_run_ids=["r1", "r2"],
            summary="All good",
        )
        d = report.to_dict()
        assert d["corpus_size"] == 10
        assert d["analysed_run_ids"] == ["r1", "r2"]
        assert d["summary"] == "All good"
        assert d["proposals"] == []

    def test_to_json_valid(self):
        import json
        report = WeaknessReport(
            proposals=[], corpus_size=1, analysed_run_ids=[], summary="x"
        )
        json.loads(report.to_json())  # must not raise


# ===========================================================================
# Individual detector: UNNECESSARY_TOOL_CALLS
# ===========================================================================


class TestDetectUnnecessaryToolCalls:
    def _make_spammy_record(self, tool_name: str, count: int) -> RunRecord:
        record = _base_record("Complex task")
        for i in range(count):
            cid = _add_tool_call(record, tool_name, {"arg": i}, call_id=f"c{i}")
            _add_tool_result(record, tool_name, cid)
        return record

    def test_no_hit_below_threshold(self):
        record = self._make_spammy_record("shell", 5)  # below threshold of 10
        hits = _detect_unnecessary_tool_calls([_entry(record)])
        assert hits == []

    def test_hit_when_high_count_and_high_ratio(self):
        record = self._make_spammy_record("shell", 12)
        hits = _detect_unnecessary_tool_calls([_entry(record)])
        assert len(hits) == 1
        assert hits[0].metrics["most_used_tool"] == "shell"
        assert hits[0].metrics["total_tool_calls"] == 12

    def test_no_hit_when_diverse_tools(self):
        record = _base_record("Multi-tool task")
        tools = ["read_file", "write_file", "edit_file", "shell", "search", "grep",
                 "diff", "run", "check", "lint", "verify"]
        for i, t in enumerate(tools):
            cid = _add_tool_call(record, t, {}, call_id=f"c{i}")
            _add_tool_result(record, t, cid)
        hits = _detect_unnecessary_tool_calls([_entry(record)])
        # Each tool used once → low repetition ratio → no hit
        assert hits == []

    def test_multiple_records_only_flagged_ones_returned(self):
        clean = self._make_spammy_record("read_file", 3)
        spammy = self._make_spammy_record("shell", 15)
        hits = _detect_unnecessary_tool_calls([_entry(clean), _entry(spammy)])
        assert len(hits) == 1
        assert hits[0].run_id == spammy.run_id


# ===========================================================================
# Individual detector: REPEATED_FAILED_APPROACHES
# ===========================================================================


class TestDetectRepeatedFailedApproaches:
    def test_no_hit_when_no_errors(self):
        record = _base_record()
        cid = _add_tool_call(record, "shell", {"command": "ls"})
        _add_tool_result(record, "shell", cid, output="file.txt")
        hits = _detect_repeated_failed_approaches([_entry(record)])
        assert hits == []

    def test_hit_when_same_tool_errors_twice(self):
        record = _base_record("Fix file")
        for i in range(3):
            cid = _add_tool_call(record, "edit_file", {"path": "f.py"}, call_id=f"e{i}")
            _add_tool_result(record, "edit_file", cid, output="Error: not found", is_error=True)
        hits = _detect_repeated_failed_approaches([_entry(record)])
        assert len(hits) == 1
        assert hits[0].metrics["failing_tool"] == "edit_file"
        assert hits[0].metrics["error_count"] == 3

    def test_no_hit_when_single_error(self):
        record = _base_record()
        cid = _add_tool_call(record, "shell", {"command": "bad"})
        _add_tool_result(record, "shell", cid, output="Error", is_error=True)
        hits = _detect_repeated_failed_approaches([_entry(record)])
        assert hits == []  # threshold is 2

    def test_only_worst_tool_reported_per_record(self):
        record = _base_record("Two bad tools")
        for i in range(2):
            cid = _add_tool_call(record, "write_file", {}, call_id=f"w{i}")
            _add_tool_result(record, "write_file", cid, output="Error", is_error=True)
        for i in range(4):
            cid = _add_tool_call(record, "shell", {}, call_id=f"s{i}")
            _add_tool_result(record, "shell", cid, output="Error", is_error=True)
        hits = _detect_repeated_failed_approaches([_entry(record)])
        assert len(hits) == 1
        assert hits[0].metrics["failing_tool"] == "shell"  # worst


# ===========================================================================
# Individual detector: POOR_VERIFICATION
# ===========================================================================


class TestDetectPoorVerification:
    def test_no_hit_when_no_write_ops(self):
        record = _base_record("Explain concept")
        hits = _detect_poor_verification([_entry(record)])
        assert hits == []

    def test_no_hit_when_verification_present(self):
        record = _base_record("Write tests")
        cid = _add_tool_call(record, "write_file", {"path": "app.py", "content": "x"})
        _add_tool_result(record, "write_file", cid)
        vcid = _add_tool_call(record, "shell", {"command": "pytest tests/"})
        _add_tool_result(record, "shell", vcid, output="1 passed")
        hits = _detect_poor_verification([_entry(record)])
        assert hits == []

    def test_hit_when_file_written_no_verification(self):
        record = _base_record("Add feature")
        cid = _add_tool_call(record, "write_file", {"path": "app.py", "content": "x"})
        _add_tool_result(record, "write_file", cid)
        record.record_final_response("Done.")
        hits = _detect_poor_verification([_entry(record)])
        assert len(hits) == 1
        assert record.run_id == hits[0].run_id

    def test_no_hit_for_failed_runs(self):
        record = _base_record("Write code")
        cid = _add_tool_call(record, "write_file", {"path": "a.py", "content": ""})
        _add_tool_result(record, "write_file", cid)
        record.outcome = RunOutcome.ERROR  # failed run — not flagged
        hits = _detect_poor_verification([_entry(record)])
        assert hits == []

    def test_hit_for_edit_file_without_test(self):
        record = _base_record("Fix bug")
        cid = _add_tool_call(record, "edit_file", {"path": "a.py", "old_str": "x", "new_str": "y"})
        _add_tool_result(record, "edit_file", cid)
        record.record_final_response("Fixed.")
        hits = _detect_poor_verification([_entry(record)])
        assert len(hits) == 1


# ===========================================================================
# Individual detector: INEFFICIENT_PLANNING
# ===========================================================================


class TestDetectInefficientPlanning:
    def _make_record_with_steps(self, steps: int) -> RunRecord:
        record = _base_record()
        record.steps_taken = steps
        return record

    def test_no_hit_when_all_similar_step_counts(self):
        entries = [_entry(self._make_record_with_steps(s)) for s in [2, 3, 3, 2, 4]]
        hits = _detect_inefficient_planning(entries)
        assert hits == []

    def test_hit_for_outlier_step_count(self):
        records = [self._make_record_with_steps(s) for s in [2, 2, 2, 2, 20]]
        entries = [_entry(r) for r in records]
        hits = _detect_inefficient_planning(entries)
        assert len(hits) == 1
        assert hits[0].metrics["steps_taken"] == 20

    def test_empty_corpus_returns_no_hits(self):
        hits = _detect_inefficient_planning([])
        assert hits == []

    def test_below_min_steps_not_flagged(self):
        records = [self._make_record_with_steps(s) for s in [1, 1, 1, 1, 3]]
        entries = [_entry(r) for r in records]
        hits = _detect_inefficient_planning(entries)
        assert hits == []  # 3 < min 4

    def test_multiple_outliers_all_returned(self):
        records = [self._make_record_with_steps(s) for s in [2, 2, 2, 30, 40]]
        entries = [_entry(r) for r in records]
        hits = _detect_inefficient_planning(entries)
        assert len(hits) == 2


# ===========================================================================
# Individual detector: BAD_TOOL_SELECTION
# ===========================================================================


class TestDetectBadToolSelection:
    def test_no_hit_when_no_shell(self):
        record = _base_record()
        cid = _add_tool_call(record, "read_file", {"path": "a.py"})
        _add_tool_result(record, "read_file", cid)
        hits = _detect_bad_tool_selection([_entry(record)])
        assert hits == []

    def test_hit_for_shell_cat(self):
        record = _base_record("Read a file")
        cid = _add_tool_call(record, "shell", {"command": "cat src/main.py"})
        _add_tool_result(record, "shell", cid, output="def main(): pass")
        hits = _detect_bad_tool_selection([_entry(record)])
        assert len(hits) == 1
        assert "cat" in hits[0].detail

    def test_hit_for_shell_echo_write(self):
        record = _base_record("Create file")
        cid = _add_tool_call(record, "shell", {"command": "echo 'content' > file.txt"})
        _add_tool_result(record, "shell", cid)
        hits = _detect_bad_tool_selection([_entry(record)])
        assert len(hits) == 1

    def test_no_hit_for_legitimate_shell(self):
        record = _base_record("Run build")
        cid = _add_tool_call(record, "shell", {"command": "make build"})
        _add_tool_result(record, "shell", cid)
        hits = _detect_bad_tool_selection([_entry(record)])
        assert hits == []

    def test_count_aggregated_per_run(self):
        record = _base_record("Multi read")
        for i, cmd in enumerate(["cat a.py", "cat b.py", "head -n 5 c.py"]):
            cid = _add_tool_call(record, "shell", {"command": cmd}, call_id=f"c{i}")
            _add_tool_result(record, "shell", cid)
        hits = _detect_bad_tool_selection([_entry(record)])
        assert len(hits) == 1
        assert hits[0].metrics["shell_file_op_count"] == 3


# ===========================================================================
# Individual detector: FAILURE_TO_RECOVER
# ===========================================================================


class TestDetectFailureToRecover:
    def test_no_hit_when_no_errors(self):
        record = _base_record()
        cid = _add_tool_call(record, "shell", {"command": "ls"}, call_id="c0")
        _add_tool_result(record, "shell", "c0", output="ok")
        cid2 = _add_tool_call(record, "shell", {"command": "ls"}, call_id="c1")
        _add_tool_result(record, "shell", "c1", output="ok")
        hits = _detect_failure_to_recover([_entry(record)])
        assert hits == []

    def test_hit_when_same_args_retried_after_error(self):
        record = _base_record("Fix it")
        # First call — error
        _add_tool_call(record, "shell", {"command": "bad_cmd"}, call_id="c0")
        _add_tool_result(record, "shell", "c0", output="Error", is_error=True)
        # Second call — identical args
        _add_tool_call(record, "shell", {"command": "bad_cmd"}, call_id="c1")
        _add_tool_result(record, "shell", "c1", output="Error", is_error=True)
        hits = _detect_failure_to_recover([_entry(record)])
        assert len(hits) == 1
        assert "shell" in hits[0].detail

    def test_no_hit_when_args_changed_after_error(self):
        record = _base_record("Adapt")
        _add_tool_call(record, "shell", {"command": "bad_cmd"}, call_id="c0")
        _add_tool_result(record, "shell", "c0", output="Error", is_error=True)
        _add_tool_call(record, "shell", {"command": "good_cmd"}, call_id="c1")
        _add_tool_result(record, "shell", "c1", output="ok")
        hits = _detect_failure_to_recover([_entry(record)])
        assert hits == []

    def test_no_hit_when_different_tool_used(self):
        record = _base_record("Switch tool")
        _add_tool_call(record, "shell", {"command": "bad_cmd"}, call_id="c0")
        _add_tool_result(record, "shell", "c0", output="Error", is_error=True)
        _add_tool_call(record, "read_file", {"path": "a.py"}, call_id="c1")
        _add_tool_result(record, "read_file", "c1", output="content")
        hits = _detect_failure_to_recover([_entry(record)])
        assert hits == []

    def test_metrics_contain_instance_count(self):
        record = _base_record("Multiple failures")
        for i in range(4):
            cid = f"c{i}"
            _add_tool_call(record, "edit_file", {"path": "f.py"}, call_id=cid)
            _add_tool_result(record, "edit_file", cid, output="Error", is_error=True)
        hits = _detect_failure_to_recover([_entry(record)])
        assert len(hits) == 1
        assert hits[0].metrics["non_recovery_count"] >= 1


# ===========================================================================
# WeaknessAnalyzer.analyse — integration tests
# ===========================================================================


class TestWeaknessAnalyzerAnalyse:
    def test_empty_corpus_returns_report_with_no_proposals(self):
        analyzer = WeaknessAnalyzer()
        report = analyzer.analyse([])
        assert report.corpus_size == 0
        assert report.proposals == []

    def test_single_clean_record_no_proposals(self):
        record = _base_record("Say hello")
        record.steps_taken = 1
        record.record_final_response("Hello!")
        analyzer = WeaknessAnalyzer()
        report = analyzer.analyse([_entry(record)])
        assert isinstance(report, WeaknessReport)
        assert report.corpus_size == 1

    def test_report_run_ids_match_entries(self):
        entries = [_entry(_base_record(f"task {i}")) for i in range(3)]
        report = WeaknessAnalyzer().analyse(entries)
        assert set(report.analysed_run_ids) == {e.run_id for e in entries}

    def test_proposals_sorted_high_before_low(self):
        # Build a corpus where poor_verification (likely HIGH) fires for most runs
        entries = []
        for _ in range(10):
            r = _base_record("Write code")
            cid = _add_tool_call(r, "write_file", {"path": "a.py", "content": "x"})
            _add_tool_result(r, "write_file", cid)
            r.record_final_response("Done")
            entries.append(_entry(r))

        # Add one run with a mildly high tool count
        spammy = _base_record("Explore")
        for i in range(12):
            cid = _add_tool_call(spammy, "shell", {"cmd": i}, call_id=f"sp{i}")
            _add_tool_result(spammy, "shell", cid)
        entries.append(_entry(spammy))

        report = WeaknessAnalyzer().analyse(entries)
        severities = [p.severity for p in report.proposals]
        order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}
        for i in range(len(severities) - 1):
            assert order[severities[i]] <= order[severities[i + 1]]

    def test_min_evidence_count_filters_rare_patterns(self):
        # Only 1 record has unnecessary tool calls — with min_evidence=2, it's filtered
        spammy = _base_record("Explore")
        for i in range(12):
            cid = _add_tool_call(spammy, "shell", {"cmd": i}, call_id=f"sp{i}")
            _add_tool_result(spammy, "shell", cid)

        analyzer = WeaknessAnalyzer(min_evidence_count=2)
        report = analyzer.analyse([_entry(spammy)])
        # unnecessary_tool_calls only has 1 hit, should be filtered
        types = [p.weakness_type for p in report.proposals]
        assert WeaknessType.UNNECESSARY_TOOL_CALLS not in types

    def test_max_evidence_per_proposal_limits_evidence_list(self):
        entries = []
        for i in range(20):
            r = _base_record(f"Write code {i}")
            cid = _add_tool_call(r, "write_file", {"path": "a.py", "content": "x"})
            _add_tool_result(r, "write_file", cid)
            r.record_final_response("Done")
            entries.append(_entry(r))

        analyzer = WeaknessAnalyzer(max_evidence_per_proposal=3)
        report = analyzer.analyse(entries)
        for proposal in report.proposals:
            assert len(proposal.evidence) <= 3

    def test_summary_mentions_severity_counts(self):
        # Create a corpus with a clear high-severity issue
        entries = []
        for _ in range(8):
            r = _base_record("Write feature")
            cid = _add_tool_call(r, "write_file", {"path": "f.py", "content": "x"})
            _add_tool_result(r, "write_file", cid)
            r.record_final_response("Done")
            entries.append(_entry(r))

        report = WeaknessAnalyzer().analyse(entries)
        assert "HIGH" in report.summary or "MEDIUM" in report.summary or "LOW" in report.summary

    def test_analyse_records_convenience_wrapper(self):
        records = [_base_record(f"t{i}") for i in range(3)]
        analyzer = WeaknessAnalyzer()
        report = analyzer.analyse_records(records)
        assert report.corpus_size == 3

    def test_analyse_records_with_evaluations(self):
        records = [_base_record(f"t{i}") for i in range(3)]
        evals = [_simple_eval() for _ in range(3)]
        report = WeaknessAnalyzer().analyse_records(records, evals)
        assert report.corpus_size == 3

    def test_proposal_fields_all_populated(self):
        entries = []
        for _ in range(5):
            r = _base_record("Write code")
            cid = _add_tool_call(r, "write_file", {"path": "a.py", "content": "x"})
            _add_tool_result(r, "write_file", cid)
            r.record_final_response("Done")
            entries.append(_entry(r))

        report = WeaknessAnalyzer().analyse(entries)
        for p in report.proposals:
            assert p.weakness_type in WeaknessType
            assert p.severity in Severity
            assert p.observed_problem
            assert p.likely_cause
            assert p.proposed_improvement
            assert p.expected_benefit
            assert p.risk
            assert isinstance(p.affected_run_count, int)
            assert 0.0 <= p.prevalence <= 1.0

    def test_failure_to_recover_detected_in_corpus(self):
        entries = []
        for _ in range(3):
            r = _base_record("Retry same error")
            _add_tool_call(r, "shell", {"command": "bad"}, call_id="c0")
            _add_tool_result(r, "shell", "c0", output="Error", is_error=True)
            _add_tool_call(r, "shell", {"command": "bad"}, call_id="c1")
            _add_tool_result(r, "shell", "c1", output="Error", is_error=True)
            entries.append(_entry(r))

        report = WeaknessAnalyzer().analyse(entries)
        types = [p.weakness_type for p in report.proposals]
        assert WeaknessType.FAILURE_TO_RECOVER in types

    def test_repeated_failed_approaches_detected_in_corpus(self):
        entries = []
        for _ in range(3):
            r = _base_record("Keep failing")
            for i in range(3):
                cid = f"ef{i}"
                _add_tool_call(r, "edit_file", {"path": "f.py"}, call_id=cid)
                _add_tool_result(r, "edit_file", cid, output="Error", is_error=True)
            entries.append(_entry(r))

        report = WeaknessAnalyzer().analyse(entries)
        types = [p.weakness_type for p in report.proposals]
        assert WeaknessType.REPEATED_FAILED_APPROACHES in types

    def test_bad_tool_selection_detected_in_corpus(self):
        entries = []
        for _ in range(3):
            r = _base_record("Read file via shell")
            cid = _add_tool_call(r, "shell", {"command": "cat src/main.py"})
            _add_tool_result(r, "shell", cid, output="content")
            entries.append(_entry(r))

        report = WeaknessAnalyzer().analyse(entries)
        types = [p.weakness_type for p in report.proposals]
        assert WeaknessType.BAD_TOOL_SELECTION in types

    def test_report_to_json_round_trips(self):
        import json
        entries = []
        for _ in range(3):
            r = _base_record("Write code")
            cid = _add_tool_call(r, "write_file", {"path": "a.py", "content": "x"})
            _add_tool_result(r, "write_file", cid)
            r.record_final_response("Done")
            entries.append(_entry(r))

        report = WeaknessAnalyzer().analyse(entries)
        data = json.loads(report.to_json())
        assert "proposals" in data
        assert "corpus_size" in data
        assert data["corpus_size"] == 3
