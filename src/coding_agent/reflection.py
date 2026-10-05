"""Reflection component for analyzing completed agent run trajectories and evaluations.

This module provides ``ReflectionResult`` and ``ReflectionGenerator`` for generating
structured retrospectives on completed agent runs using a language model.
Reflection is kept strictly decoupled from execution and evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ReflectionResult:
    """Structured retrospective reflection on an agent execution run.

    Attributes
    ----------
    what_worked:
        Summary of strategies, tool calls, or actions that succeeded.
    what_failed:
        Summary of errors, failed tool calls, or bottlenecks encountered.
    why_it_failed:
        Root-cause analysis explaining why failures or issues occurred.
    what_to_do_differently:
        Actionable recommendations and alternative approaches for future runs.
    summary:
        High-level concise reflective summary of the run.
    raw_response:
        Raw model response text before parsing (if available).
    """

    what_worked: str
    what_failed: str
    why_it_failed: str
    what_to_do_differently: str
    summary: str = ""
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise ReflectionResult to a plain, JSON-safe dictionary."""
        return {
            "what_worked": self.what_worked,
            "what_failed": self.what_failed,
            "why_it_failed": self.why_it_failed,
            "what_to_do_differently": self.what_to_do_differently,
            "summary": self.summary,
            "raw_response": self.raw_response,
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialise ReflectionResult to a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReflectionResult":
        """Reconstruct ReflectionResult from a dictionary representation."""
        return cls(
            what_worked=str(data.get("what_worked", "")),
            what_failed=str(data.get("what_failed", "")),
            why_it_failed=str(data.get("why_it_failed", "")),
            what_to_do_differently=str(data.get("what_to_do_differently", "")),
            summary=str(data.get("summary", "")),
            raw_response=data.get("raw_response"),
        )

    @classmethod
    def from_json(cls, text: str) -> "ReflectionResult":
        """Reconstruct ReflectionResult from a JSON string."""
        return cls.from_dict(json.loads(text))
