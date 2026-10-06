"""Long-term experience memory for the terminal coding agent.

This module provides ``Experience`` and ``ExperienceMemory`` for storing and
retrieving past agent execution experiences (task prompt, evaluation result, and
reflection analysis).  Retrieval uses term-matching and evaluation scores without
requiring an external vector database. Memory remains completely independent from
the core agent execution loop.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from coding_agent.evaluator import EvaluationResult
from coding_agent.reflection import ReflectionResult


@dataclass
class Experience:
    """Structured record of one completed agent experience.

    Attributes
    ----------
    task:
        The initial task prompt or problem description.
    evaluation:
        The deterministic EvaluationResult associated with the run.
    reflection:
        The structured ReflectionResult summarizing lessons learned.
    experience_id:
        Unique identifier for this experience. Auto-generated as UUID4.
    trajectory_run_id:
        Optional UUID referencing the original RunRecord trajectory.
    created_at:
        UTC timestamp when this experience record was created.
    metadata:
        Arbitrary extra metadata tags or key/value attributes.
    """

    task: str
    evaluation: EvaluationResult
    reflection: ReflectionResult
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trajectory_run_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise Experience to a plain, JSON-safe dictionary."""
        return {
            "experience_id": self.experience_id,
            "task": self.task,
            "evaluation": self.evaluation.to_dict(),
            "reflection": self.reflection.to_dict(),
            "trajectory_run_id": self.trajectory_run_id,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialise Experience to a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        """Reconstruct an Experience from its dictionary representation."""
        created_str = data.get("created_at") or data.get("started_at")
        created_dt = (
            datetime.fromisoformat(created_str)
            if created_str
            else datetime.now(timezone.utc)
        )
        return cls(
            experience_id=str(data["experience_id"]),
            task=str(data.get("task", "")),
            evaluation=EvaluationResult.from_dict(data["evaluation"]),
            reflection=ReflectionResult.from_dict(data["reflection"]),
            trajectory_run_id=data.get("trajectory_run_id"),
            created_at=created_dt,
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> "Experience":
        """Reconstruct an Experience from a JSON string."""
        return cls.from_dict(json.loads(text))
