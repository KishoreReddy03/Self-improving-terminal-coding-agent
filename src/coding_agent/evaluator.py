"""Deterministic evaluator for completed agent execution trajectories.

This module provides ``TrajectoryEvaluator`` and ``EvaluationResult`` for assessing
run trajectories using objective, rule-based signals such as completion status,
tool call success rates, step iteration limits, and shell verification command
results.  No LLM calls or non-deterministic judges are used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class EvaluationResult:
    """Structured evaluation result for an agent execution trajectory.

    Attributes
    ----------
    success:
        Overall pass/fail status of the trajectory according to deterministic criteria.
    score:
        Quantitative score between 0.0 and 1.0.
    reason:
        Human-readable summary explanation of the evaluation outcome and score.
    details:
        Structured breakdown of individual signals evaluated.
    """

    success: bool
    score: float
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise EvaluationResult to a plain, JSON-safe dictionary."""
        return {
            "success": self.success,
            "score": self.score,
            "reason": self.reason,
            "details": dict(self.details),
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialise EvaluationResult to a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Reconstruct EvaluationResult from dictionary representation."""
        return cls(
            success=bool(data["success"]),
            score=float(data["score"]),
            reason=str(data["reason"]),
            details=dict(data.get("details") or {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "EvaluationResult":
        """Reconstruct EvaluationResult from JSON string."""
        return cls.from_dict(json.loads(text))


from coding_agent.run_record import RunEvent, RunEventType, RunOutcome, RunRecord


class TrajectoryEvaluator:
    """Evaluates completed agent trajectories using deterministic signals."""

    VERIFICATION_KEYWORDS: Tuple[str, ...] = (
        "pytest",
        "unittest",
        "test",
        "npm test",
        "yarn test",
        "cargo test",
        "go test",
        "make test",
        "check",
        "lint",
        "mypy",
        "flake8",
        "ruff",
        "coverage",
    )

    def _analyze_verification_commands(
        self, record: RunRecord
    ) -> Tuple[bool, Optional[bool], List[Dict[str, Any]]]:
        """Inspect shell tool calls to detect and evaluate verification commands."""
        tool_call_map: Dict[str, str] = {}
        for event in record.events:
            if event.event_type == RunEventType.TOOL_CALL:
                tool_name = event.data.get("tool_name", "")
                call_id = event.data.get("tool_call_id", "")
                args = event.data.get("arguments", {})
                if (
                    "shell" in tool_name
                    or "command" in tool_name
                    or "command" in args
                ):
                    cmd_str = args.get("command", "") if isinstance(args, dict) else ""
                    tool_call_map[call_id] = cmd_str

        verification_details: List[Dict[str, Any]] = []
        for event in record.events:
            if event.event_type == RunEventType.TOOL_RESULT:
                call_id = event.data.get("tool_call_id", "")
                if call_id in tool_call_map:
                    cmd = tool_call_map[call_id]
                    is_verif = any(kw in cmd.lower() for kw in self.VERIFICATION_KEYWORDS)
                    output_str = event.data.get("output", "")
                    is_error = event.data.get("is_error", False)

                    exit_code = 0 if not is_error else -1
                    if isinstance(output_str, str) and output_str.strip().startswith("{"):
                        try:
                            parsed_out = json.loads(output_str)
                            if isinstance(parsed_out, dict) and "exit_code" in parsed_out:
                                exit_code = int(parsed_out["exit_code"])
                                is_error = exit_code != 0
                        except Exception:
                            pass

                    cmd_succeeded = not is_error and exit_code == 0

                    verification_details.append(
                        {
                            "command": cmd,
                            "is_verification_cmd": is_verif,
                            "succeeded": cmd_succeeded,
                            "tool_call_id": call_id,
                        }
                    )

        if not verification_details:
            return False, None, []

        explicit_verifs = [v for v in verification_details if v["is_verification_cmd"]]
        target_list = explicit_verifs if explicit_verifs else verification_details

        last_cmd = target_list[-1]
        return True, last_cmd["succeeded"], verification_details
