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
