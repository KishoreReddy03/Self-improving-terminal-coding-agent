"""Weakness analyzer for recurring agent execution problems.

This module provides ``WeaknessAnalyzer`` — a corpus-level inspector that
examines a collection of evaluated trajectories and identifies *repeated*
behavioural problems.  It operates entirely on ``RunRecord`` and
``EvaluationResult`` data; it never calls an LLM and never modifies any
agent state.

Detected weakness categories
-----------------------------
- **UNNECESSARY_TOOL_CALLS**   – high tool-call counts relative to task
  complexity or with a high proportion of redundant same-tool repetitions.
- **REPEATED_FAILED_APPROACHES** – the same tool is called multiple times in
  the same run and consistently produces errors.
- **POOR_VERIFICATION**        – code-writing runs that never run a
  verification command (test / lint / check).
- **INEFFICIENT_PLANNING**     – runs use many more steps than peers for
  similar task types (high-step outliers).
- **BAD_TOOL_SELECTION**       – shell is used to do things a purpose-built
  tool (read_file, write_file, edit_file) could do more safely.
- **FAILURE_TO_RECOVER**       – a tool error is followed immediately by
  calling the *same* tool with identical arguments (no adaptation).

Design constraints
-------------------
- No LLM calls — all analysis is deterministic.
- Read-only — never writes to disk, never touches the agent.
- The analyzer produces proposals; applying them is a separate step.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from coding_agent.evaluator import EvaluationResult
from coding_agent.run_record import RunEvent, RunEventType, RunOutcome, RunRecord


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class WeaknessType(str, Enum):
    """Canonical label for each class of detectable weakness."""

    UNNECESSARY_TOOL_CALLS = "unnecessary_tool_calls"
    REPEATED_FAILED_APPROACHES = "repeated_failed_approaches"
    POOR_VERIFICATION = "poor_verification"
    INEFFICIENT_PLANNING = "inefficient_planning"
    BAD_TOOL_SELECTION = "bad_tool_selection"
    FAILURE_TO_RECOVER = "failure_to_recover"


class Severity(str, Enum):
    """How severe and how frequently a weakness pattern was observed."""

    LOW = "low"       # Observed in <20 % of relevant runs
    MEDIUM = "medium" # Observed in 20–50 % of relevant runs
    HIGH = "high"     # Observed in >50 % of relevant runs


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class WeaknessEvidence:
    """A single trajectory-level piece of evidence for a weakness.

    Attributes
    ----------
    run_id:
        The unique ID of the ``RunRecord`` where the problem was observed.
    task_snippet:
        First 120 characters of the task prompt for quick human scanning.
    detail:
        Concise description of the specific problem instance found in this run.
    metrics:
        Optional numeric data supporting the observation (counts, ratios, …).
    """

    run_id: str
    task_snippet: str
    detail: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_snippet": self.task_snippet,
            "detail": self.detail,
            "metrics": self.metrics,
        }


@dataclass
class ImprovementProposal:
    """Structured proposal for addressing an observed recurring weakness.

    Attributes
    ----------
    weakness_type:
        The canonical weakness category this proposal addresses.
    severity:
        Estimated impact level based on frequency and performance effect.
    observed_problem:
        Plain-language description of the behavioural problem detected.
    evidence:
        List of trajectory-level evidence instances supporting the claim.
    affected_run_count:
        How many trajectories in the analysed corpus show this weakness.
    corpus_size:
        Total number of trajectories analysed.
    prevalence:
        Fraction of corpus runs where this weakness was observed (0.0–1.0).
    likely_cause:
        Hypothesis about the root cause of the problem.
    proposed_improvement:
        Actionable recommendation for how the agent or its prompts should change.
    expected_benefit:
        What specific metric or behaviour should improve if the change is applied.
    risk:
        Potential downside or tradeoff of applying the proposed change.
    """

    weakness_type: WeaknessType
    severity: Severity
    observed_problem: str
    evidence: List[WeaknessEvidence]
    affected_run_count: int
    corpus_size: int
    prevalence: float
    likely_cause: str
    proposed_improvement: str
    expected_benefit: str
    risk: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain, JSON-safe dictionary."""
        return {
            "weakness_type": self.weakness_type.value,
            "severity": self.severity.value,
            "observed_problem": self.observed_problem,
            "evidence": [e.to_dict() for e in self.evidence],
            "affected_run_count": self.affected_run_count,
            "corpus_size": self.corpus_size,
            "prevalence": round(self.prevalence, 3),
            "likely_cause": self.likely_cause,
            "proposed_improvement": self.proposed_improvement,
            "expected_benefit": self.expected_benefit,
            "risk": self.risk,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass
class WeaknessReport:
    """Aggregated output of a full corpus weakness analysis.

    Attributes
    ----------
    proposals:
        All ``ImprovementProposal`` objects found, sorted by severity then
        prevalence (highest-impact first).
    corpus_size:
        Number of trajectories included in the analysis.
    analysed_run_ids:
        Run IDs of all trajectories that were examined.
    summary:
        High-level narrative paragraph describing the most important findings.
    """

    proposals: List[ImprovementProposal]
    corpus_size: int
    analysed_run_ids: List[str]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corpus_size": self.corpus_size,
            "analysed_run_ids": self.analysed_run_ids,
            "summary": self.summary,
            "proposals": [p.to_dict() for p in self.proposals],
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


# ---------------------------------------------------------------------------
# Trajectory pair helper
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryEntry:
    """A pair of RunRecord and its optional EvaluationResult."""

    record: RunRecord
    evaluation: Optional[EvaluationResult] = None

    @property
    def run_id(self) -> str:
        return self.record.run_id

    @property
    def task_snippet(self) -> str:
        return (self.record.task or "")[:120]


# ---------------------------------------------------------------------------
# Individual detector functions
# ---------------------------------------------------------------------------

# Threshold constants — adjust without touching detector logic
_UNNECESSARY_TOOL_HIGH_THRESHOLD = 10    # ≥ this many tool calls in a run
_UNNECESSARY_TOOL_RATIO_THRESHOLD = 0.40 # ≥ this fraction of calls repeat the same tool
_FAILED_APPROACH_REPEAT_THRESHOLD = 2    # same tool errors ≥ this many times in one run
_PLANNING_STEP_OUTLIER_MULTIPLIER = 2.0  # steps ≥ 2× median considered inefficient
_PLANNING_MIN_STEPS = 4                  # ignore very short runs for this check
_FILE_OP_SHELL_PATTERNS = (              # shell commands that a file tool should handle
    "cat ", "echo ", "touch ", "mkdir", "cp ", "mv ", "sed ", "awk ",
    "head ", "tail ", "grep ", "tee ",
)
_VERIFICATION_WRITE_TOOLS = ("write_file", "edit_file", "shell")  # suggest verification after these
_VERIFICATION_KEYWORDS = (
    "pytest", "unittest", "test", "npm test", "yarn test", "cargo test",
    "go test", "make test", "check", "lint", "mypy", "flake8", "ruff",
    "coverage",
)


def _tool_calls_by_name(record: RunRecord) -> Dict[str, List[RunEvent]]:
    """Group TOOL_CALL events by tool_name."""
    groups: Dict[str, List[RunEvent]] = defaultdict(list)
    for ev in record.tool_calls:
        groups[ev.data.get("tool_name", "unknown")].append(ev)
    return dict(groups)


def _error_results_by_tool(record: RunRecord) -> Dict[str, int]:
    """Count error TOOL_RESULT events per tool_name."""
    counts: Counter = Counter()
    for ev in record.error_tool_results():
        counts[ev.data.get("tool_name", "unknown")] += 1
    return dict(counts)


def _detect_unnecessary_tool_calls(
    entries: List[TrajectoryEntry],
) -> List[WeaknessEvidence]:
    """Detect runs with excessive or highly redundant tool call patterns."""
    hits: List[WeaknessEvidence] = []
    for entry in entries:
        record = entry.record
        total = record.tool_call_count()
        if total < _UNNECESSARY_TOOL_HIGH_THRESHOLD:
            continue

        by_name = _tool_calls_by_name(record)
        if not by_name:
            continue

        most_common_tool, most_common_events = max(
            by_name.items(), key=lambda x: len(x[1])
        )
        most_common_count = len(most_common_events)
        ratio = most_common_count / total

        if ratio >= _UNNECESSARY_TOOL_RATIO_THRESHOLD:
            hits.append(
                WeaknessEvidence(
                    run_id=entry.run_id,
                    task_snippet=entry.task_snippet,
                    detail=(
                        f"{total} total tool calls; '{most_common_tool}' called "
                        f"{most_common_count} times ({ratio:.0%} of all calls)."
                    ),
                    metrics={
                        "total_tool_calls": total,
                        "most_used_tool": most_common_tool,
                        "most_used_count": most_common_count,
                        "repetition_ratio": round(ratio, 3),
                    },
                )
            )
    return hits


def _detect_repeated_failed_approaches(
    entries: List[TrajectoryEntry],
) -> List[WeaknessEvidence]:
    """Detect runs where the same tool fails multiple times with no change."""
    hits: List[WeaknessEvidence] = []
    for entry in entries:
        record = entry.record
        error_counts = _error_results_by_tool(record)
        bad_tools = {
            t: c
            for t, c in error_counts.items()
            if c >= _FAILED_APPROACH_REPEAT_THRESHOLD
        }
        if bad_tools:
            worst_tool = max(bad_tools, key=lambda t: bad_tools[t])
            worst_count = bad_tools[worst_tool]
            hits.append(
                WeaknessEvidence(
                    run_id=entry.run_id,
                    task_snippet=entry.task_snippet,
                    detail=(
                        f"Tool '{worst_tool}' produced {worst_count} consecutive "
                        f"error results without a successful recovery."
                    ),
                    metrics={
                        "failing_tool": worst_tool,
                        "error_count": worst_count,
                        "all_error_tools": bad_tools,
                    },
                )
            )
    return hits


def _run_has_verification(record: RunRecord) -> bool:
    """Return True if the run contains at least one verification command."""
    for ev in record.tool_results:
        # Check against associated tool_call to find command text
        pass

    # Check TOOL_CALL events for verification keywords in shell commands
    for ev in record.tool_calls:
        tool_name = ev.data.get("tool_name", "")
        if "shell" in tool_name or "command" in tool_name:
            cmd = ev.data.get("arguments", {}).get("command", "")
            if any(kw in cmd.lower() for kw in _VERIFICATION_KEYWORDS):
                return True
    return False


def _run_has_write_operations(record: RunRecord) -> bool:
    """Return True if the run wrote or edited any files."""
    for ev in record.tool_calls:
        tool_name = ev.data.get("tool_name", "")
        if tool_name in ("write_file", "edit_file", "WriteFileTool", "EditFileTool"):
            return True
        # Also detect shell-based writes
        if "shell" in tool_name or "command" in tool_name:
            cmd = ev.data.get("arguments", {}).get("command", "")
            # Heuristic: shell redirections and file-writing patterns
            if any(p in cmd for p in (">", "tee ", "echo ", "cat >")):
                return True
    return False


def _detect_poor_verification(
    entries: List[TrajectoryEntry],
) -> List[WeaknessEvidence]:
    """Detect runs that write code or files but never verify the result."""
    hits: List[WeaknessEvidence] = []
    for entry in entries:
        record = entry.record
        if not _run_has_write_operations(record):
            continue
        if _run_has_verification(record):
            continue
        # Only flag runs that completed (incomplete runs have other issues)
        if record.outcome != RunOutcome.SUCCESS:
            continue
        hits.append(
            WeaknessEvidence(
                run_id=entry.run_id,
                task_snippet=entry.task_snippet,
                detail=(
                    "Run wrote/edited files but ran no verification command "
                    "(pytest, lint, check, etc.)."
                ),
                metrics={
                    "write_ops": sum(
                        1 for ev in record.tool_calls
                        if ev.data.get("tool_name", "") in (
                            "write_file", "edit_file", "WriteFileTool", "EditFileTool"
                        )
                    ),
                },
            )
        )
    return hits


def _corpus_step_median(entries: List[TrajectoryEntry]) -> float:
    """Compute median steps_taken across the corpus (ignore zero-step runs)."""
    steps = [e.record.steps_taken for e in entries if e.record.steps_taken > 0]
    if not steps:
        return 0.0
    steps.sort()
    mid = len(steps) // 2
    return float(steps[mid] if len(steps) % 2 else (steps[mid - 1] + steps[mid]) / 2)


def _detect_inefficient_planning(
    entries: List[TrajectoryEntry],
) -> List[WeaknessEvidence]:
    """Detect runs with disproportionately many steps relative to peers."""
    median = _corpus_step_median(entries)
    if median < 1:
        return []
    threshold = max(_PLANNING_MIN_STEPS, median * _PLANNING_STEP_OUTLIER_MULTIPLIER)

    hits: List[WeaknessEvidence] = []
    for entry in entries:
        steps = entry.record.steps_taken
        if steps < threshold:
            continue
        hits.append(
            WeaknessEvidence(
                run_id=entry.run_id,
                task_snippet=entry.task_snippet,
                detail=(
                    f"Run used {steps} steps (corpus median: {median:.1f}; "
                    f"outlier threshold: {threshold:.1f})."
                ),
                metrics={
                    "steps_taken": steps,
                    "corpus_median": round(median, 1),
                    "threshold": round(threshold, 1),
                },
            )
        )
    return hits


def _detect_bad_tool_selection(
    entries: List[TrajectoryEntry],
) -> List[WeaknessEvidence]:
    """Detect runs that use shell for file operations a dedicated tool should handle."""
    hits: List[WeaknessEvidence] = []
    for entry in entries:
        record = entry.record
        shell_file_ops: List[str] = []

        for ev in record.tool_calls:
            tool_name = ev.data.get("tool_name", "")
            if "shell" not in tool_name and "command" not in tool_name:
                continue
            cmd = str(ev.data.get("arguments", {}).get("command", "")).strip()
            matched = [p for p in _FILE_OP_SHELL_PATTERNS if cmd.startswith(p.strip()) or f" {p.strip()} " in cmd]
            if matched:
                shell_file_ops.append(cmd[:80])

        if shell_file_ops:
            hits.append(
                WeaknessEvidence(
                    run_id=entry.run_id,
                    task_snippet=entry.task_snippet,
                    detail=(
                        f"Shell used for {len(shell_file_ops)} file-system operation(s) "
                        f"that dedicated tools could handle more safely. "
                        f"Sample: {shell_file_ops[0]!r}"
                    ),
                    metrics={
                        "shell_file_op_count": len(shell_file_ops),
                        "sample_commands": shell_file_ops[:3],
                    },
                )
            )
    return hits


def _detect_failure_to_recover(
    entries: List[TrajectoryEntry],
) -> List[WeaknessEvidence]:
    """Detect runs where the same tool is re-called with identical args after an error."""
    hits: List[WeaknessEvidence] = []

    for entry in entries:
        record = entry.record
        # Build a map from tool_call_id → result is_error
        result_map: Dict[str, bool] = {}
        for ev in record.tool_results:
            result_map[ev.data.get("tool_call_id", "")] = ev.data.get("is_error", False)

        # Walk tool calls in order; track last call per tool with its args
        last_call: Dict[str, Tuple[str, Any]] = {}  # tool_name → (call_id, args_json)
        non_recovering_instances: List[str] = []

        for ev in record.tool_calls:
            tool_name = ev.data.get("tool_name", "unknown")
            call_id = ev.data.get("tool_call_id", "")
            args = ev.data.get("arguments", {})
            args_key = json.dumps(args, sort_keys=True)

            if tool_name in last_call:
                prev_call_id, prev_args_key = last_call[tool_name]
                prev_was_error = result_map.get(prev_call_id, False)
                if prev_was_error and args_key == prev_args_key:
                    non_recovering_instances.append(
                        f"'{tool_name}' re-called with identical args after error "
                        f"(call_id={call_id})"
                    )

            last_call[tool_name] = (call_id, args_key)

        if non_recovering_instances:
            hits.append(
                WeaknessEvidence(
                    run_id=entry.run_id,
                    task_snippet=entry.task_snippet,
                    detail=(
                        f"{len(non_recovering_instances)} non-recovery instance(s): "
                        f"{non_recovering_instances[0]}"
                    ),
                    metrics={
                        "non_recovery_count": len(non_recovering_instances),
                        "instances": non_recovering_instances,
                    },
                )
            )
    return hits


# ---------------------------------------------------------------------------
# Severity computation
# ---------------------------------------------------------------------------


def _severity(prevalence: float) -> Severity:
    if prevalence >= 0.5:
        return Severity.HIGH
    if prevalence >= 0.2:
        return Severity.MEDIUM
    return Severity.LOW


# ---------------------------------------------------------------------------
# Proposal builders
# ---------------------------------------------------------------------------

_PROPOSALS: Dict[WeaknessType, Dict[str, str]] = {
    WeaknessType.UNNECESSARY_TOOL_CALLS: {
        "likely_cause": (
            "The agent may be stuck in a loop of exploratory tool calls without "
            "a clear stopping criterion, or the system prompt does not encourage "
            "concise, targeted tool use."
        ),
        "proposed_improvement": (
            "Add a system-prompt directive to prefer reading context once and "
            "acting decisively; introduce a per-run tool-call budget signal that "
            "the agent sees in its prompt once it exceeds a soft limit."
        ),
        "expected_benefit": (
            "Reduced average tool calls per run, lower latency, and lower "
            "cost per task completion."
        ),
        "risk": (
            "Setting too aggressive a budget could cause the agent to under-explore "
            "genuinely complex tasks that require many legitimate tool calls."
        ),
    },
    WeaknessType.REPEATED_FAILED_APPROACHES: {
        "likely_cause": (
            "After a tool error the agent does not sufficiently vary its approach. "
            "It may lack explicit instruction to diagnose the error before retrying, "
            "or the error output is not informative enough to guide adaptation."
        ),
        "proposed_improvement": (
            "Add a system-prompt rule: 'If a tool call fails, read its error output "
            "carefully before retrying. Never retry with identical arguments. "
            "Consider an alternative approach or tool after two consecutive failures "
            "from the same tool.'"
        ),
        "expected_benefit": (
            "Fewer wasted tool calls, faster error recovery, higher task completion "
            "rates in runs that hit transient or fixable errors."
        ),
        "risk": (
            "Overly strict retry rules might prevent legitimate retries (e.g. "
            "transient network errors where the same command is correct)."
        ),
    },
    WeaknessType.POOR_VERIFICATION: {
        "likely_cause": (
            "The agent treats task completion as writing the code rather than "
            "confirming it works. Verification is not part of the agent's inferred "
            "definition of 'done'."
        ),
        "proposed_improvement": (
            "Amend the system prompt with: 'After writing or editing any code, "
            "always run at least one verification command (pytest, lint, or the "
            "project's test suite) before responding that the task is complete.'"
        ),
        "expected_benefit": (
            "Higher rate of runs that include passing test evidence, reducing "
            "silent regressions delivered to users."
        ),
        "risk": (
            "Some tasks genuinely have no testable output (e.g. writing documentation). "
            "The rule must be applied only when a test suite or linter is available."
        ),
    },
    WeaknessType.INEFFICIENT_PLANNING: {
        "likely_cause": (
            "The agent may be planning excessively before acting, or struggling to "
            "navigate unfamiliar codebases and spending many steps in exploration "
            "before making progress. It may also be re-reading already-seen context."
        ),
        "proposed_improvement": (
            "Encourage the agent to form a brief explicit plan (1-3 bullet points) "
            "before starting tool use, then execute it directly. Add a system-prompt "
            "reminder to avoid re-reading files already in context."
        ),
        "expected_benefit": (
            "Reduced step counts for comparable tasks; lower latency and cost."
        ),
        "risk": (
            "Enforcing strict planning could reduce flexibility for novel or "
            "ambiguous tasks that genuinely require exploratory back-and-forth."
        ),
    },
    WeaknessType.BAD_TOOL_SELECTION: {
        "likely_cause": (
            "The agent defaults to shell commands for file operations because shell "
            "is familiar and general-purpose. It may not prioritise safer, more "
            "structured tools when both are available."
        ),
        "proposed_improvement": (
            "Update the system prompt with a tool-preference hierarchy: "
            "'Prefer read_file/write_file/edit_file over shell for all file "
            "read/write operations. Use shell only when no dedicated tool exists "
            "for the operation.'"
        ),
        "expected_benefit": (
            "More predictable file operations, fewer shell injection risks, "
            "better error messages from purpose-built tools."
        ),
        "risk": (
            "Some file operations (e.g. bulk renames, chmod) have no dedicated "
            "tool and genuinely require shell. Over-restricting shell use could "
            "block legitimate operations."
        ),
    },
    WeaknessType.FAILURE_TO_RECOVER: {
        "likely_cause": (
            "The agent does not maintain explicit awareness of which tool calls have "
            "already failed. Without this state, it reissues the same call hoping "
            "for a different result — a form of blind retry."
        ),
        "proposed_improvement": (
            "Add a system-prompt instruction: 'Before any tool call, check if you "
            "have already called this tool with these exact arguments and it failed. "
            "If so, change your approach: try a different tool, modify the arguments, "
            "or ask for clarification.'"
        ),
        "expected_benefit": (
            "Faster exit from dead-end approaches; higher rate of successful "
            "error recovery leading to task completion."
        ),
        "risk": (
            "Very strict deduplication could prevent valid retries for "
            "idempotent operations where success is not guaranteed on first call "
            "(e.g. polling for a build to finish)."
        ),
    },
}


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------


class WeaknessAnalyzer:
    """Analyse a corpus of evaluated trajectories for recurring weaknesses.

    Parameters
    ----------
    min_evidence_count:
        Minimum number of trajectory hits required for a weakness to produce
        a proposal.  Proposals with fewer instances than this are silently
        dropped.  Defaults to 1 (include everything).
    max_evidence_per_proposal:
        Maximum number of trajectory evidence items to include in each proposal
        (oldest hits are trimmed).  Keeps reports readable.
    """

    def __init__(
        self,
        min_evidence_count: int = 1,
        max_evidence_per_proposal: int = 5,
    ) -> None:
        self.min_evidence_count = max(1, min_evidence_count)
        self.max_evidence_per_proposal = max(1, max_evidence_per_proposal)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        entries: Sequence[TrajectoryEntry],
    ) -> WeaknessReport:
        """Analyse a corpus of trajectories and return a WeaknessReport.

        Parameters
        ----------
        entries:
            Sequence of ``TrajectoryEntry`` objects (RunRecord + optional
            EvaluationResult).  At least 1 entry is required.

        Returns
        -------
        WeaknessReport
            Structured report containing improvement proposals sorted by
            severity, plus a high-level summary.
        """
        entries = list(entries)
        corpus_size = len(entries)
        run_ids = [e.run_id for e in entries]

        detectors = [
            (WeaknessType.UNNECESSARY_TOOL_CALLS, _detect_unnecessary_tool_calls),
            (WeaknessType.REPEATED_FAILED_APPROACHES, _detect_repeated_failed_approaches),
            (WeaknessType.POOR_VERIFICATION, _detect_poor_verification),
            (WeaknessType.INEFFICIENT_PLANNING, _detect_inefficient_planning),
            (WeaknessType.BAD_TOOL_SELECTION, _detect_bad_tool_selection),
            (WeaknessType.FAILURE_TO_RECOVER, _detect_failure_to_recover),
        ]

        proposals: List[ImprovementProposal] = []
        for weakness_type, detector in detectors:
            evidence_list = detector(entries)
            if len(evidence_list) < self.min_evidence_count:
                continue

            # Trim to max_evidence_per_proposal most-recent items
            trimmed = evidence_list[: self.max_evidence_per_proposal]
            affected = len(evidence_list)
            prevalence = affected / corpus_size if corpus_size else 0.0
            severity = _severity(prevalence)

            meta = _PROPOSALS[weakness_type]
            observed_problem = self._describe_problem(weakness_type, evidence_list, corpus_size)

            proposals.append(
                ImprovementProposal(
                    weakness_type=weakness_type,
                    severity=severity,
                    observed_problem=observed_problem,
                    evidence=trimmed,
                    affected_run_count=affected,
                    corpus_size=corpus_size,
                    prevalence=prevalence,
                    likely_cause=meta["likely_cause"],
                    proposed_improvement=meta["proposed_improvement"],
                    expected_benefit=meta["expected_benefit"],
                    risk=meta["risk"],
                )
            )

        # Sort: HIGH → MEDIUM → LOW, then by prevalence descending
        severity_order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}
        proposals.sort(key=lambda p: (severity_order[p.severity], -p.prevalence))

        summary = self._build_summary(proposals, corpus_size)

        return WeaknessReport(
            proposals=proposals,
            corpus_size=corpus_size,
            analysed_run_ids=run_ids,
            summary=summary,
        )

    def analyse_records(
        self,
        records: Sequence[RunRecord],
        evaluations: Optional[Sequence[Optional[EvaluationResult]]] = None,
    ) -> WeaknessReport:
        """Convenience wrapper: accept raw RunRecords instead of TrajectoryEntry objects.

        Parameters
        ----------
        records:
            Sequence of RunRecord objects.
        evaluations:
            Optional parallel sequence of EvaluationResult objects.  Must be
            the same length as *records* if provided.  ``None`` entries are
            allowed for runs that were not evaluated.
        """
        if evaluations is None:
            evaluations = [None] * len(records)
        entries = [
            TrajectoryEntry(record=r, evaluation=e)
            for r, e in zip(records, evaluations)
        ]
        return self.analyse(entries)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _describe_problem(
        weakness_type: WeaknessType,
        evidence: List[WeaknessEvidence],
        corpus_size: int,
    ) -> str:
        """Build a concise observed-problem description with corpus statistics."""
        count = len(evidence)
        pct = int(round(100 * count / corpus_size)) if corpus_size else 0

        descriptions = {
            WeaknessType.UNNECESSARY_TOOL_CALLS: (
                f"In {count} of {corpus_size} runs ({pct}%), the agent made an "
                f"unusually high number of tool calls with significant repetition "
                f"of the same tool, suggesting unnecessary or redundant operations."
            ),
            WeaknessType.REPEATED_FAILED_APPROACHES: (
                f"In {count} of {corpus_size} runs ({pct}%), the same tool "
                f"produced multiple consecutive errors without the agent successfully "
                f"changing its approach."
            ),
            WeaknessType.POOR_VERIFICATION: (
                f"In {count} of {corpus_size} runs ({pct}%), the agent wrote or "
                f"edited code files but completed the task without running any "
                f"verification commands (tests, linters, or checks)."
            ),
            WeaknessType.INEFFICIENT_PLANNING: (
                f"In {count} of {corpus_size} runs ({pct}%), the agent took "
                f"significantly more steps than the corpus median, suggesting "
                f"excessive exploration or redundant planning."
            ),
            WeaknessType.BAD_TOOL_SELECTION: (
                f"In {count} of {corpus_size} runs ({pct}%), the agent used shell "
                f"commands for file operations that could have been handled more "
                f"safely by read_file, write_file, or edit_file tools."
            ),
            WeaknessType.FAILURE_TO_RECOVER: (
                f"In {count} of {corpus_size} runs ({pct}%), the agent re-called a "
                f"tool with identical arguments immediately after that same call had "
                f"returned an error — showing no adaptive recovery."
            ),
        }
        return descriptions.get(weakness_type, f"Weakness '{weakness_type.value}' detected in {count} runs.")

    @staticmethod
    def _build_summary(proposals: List[ImprovementProposal], corpus_size: int) -> str:
        """Generate a plain-text executive summary of the analysis."""
        if not proposals:
            return (
                f"Analysis of {corpus_size} trajectory(s) found no recurring "
                f"weakness patterns meeting the minimum evidence threshold."
            )

        high = [p for p in proposals if p.severity == Severity.HIGH]
        medium = [p for p in proposals if p.severity == Severity.MEDIUM]
        low = [p for p in proposals if p.severity == Severity.LOW]

        lines = [
            f"Weakness analysis across {corpus_size} trajectory(s) identified "
            f"{len(proposals)} recurring problem(s): "
            f"{len(high)} HIGH, {len(medium)} MEDIUM, {len(low)} LOW severity.",
        ]

        if high:
            names = ", ".join(p.weakness_type.value for p in high)
            lines.append(
                f"HIGH-severity issues ({names}) affect more than half of "
                f"analysed runs and should be addressed with the highest priority."
            )
        if medium:
            names = ", ".join(p.weakness_type.value for p in medium)
            lines.append(
                f"MEDIUM-severity issues ({names}) affect 20–50% of runs "
                f"and represent significant improvement opportunities."
            )
        top = proposals[0]
        lines.append(
            f"Top priority: '{top.weakness_type.value}' "
            f"(prevalence {top.prevalence:.0%}, severity {top.severity.value}). "
            f"Proposed fix: {top.proposed_improvement[:120]}…"
        )

        return " ".join(lines)
