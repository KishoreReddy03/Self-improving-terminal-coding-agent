"""Long-term experience memory for the terminal coding agent.

This module provides ``Experience`` and ``ExperienceMemory`` for storing and
retrieving past agent execution experiences (task prompt, evaluation result, and
reflection analysis).  Retrieval uses term-matching and evaluation scores without
requiring an external vector database. Memory remains completely independent from
the core agent execution loop.
"""

from __future__ import annotations

import json
import re
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


class ExperienceMemory:
    """Local file-backed long-term memory store for agent experiences."""

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None) -> None:
        if storage_dir is None:
            self.storage_dir = Path.home() / ".coding_agent" / "memory"
        else:
            self.storage_dir = Path(storage_dir)

    def _ensure_directory(self) -> None:
        """Create storage directory if missing."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, experience_id: str) -> Path:
        """Return expected file path for a given experience_id."""
        return self.storage_dir / f"{experience_id}.json"

    def save(self, experience: Experience) -> Path:
        """Persist an Experience instance to local JSON storage."""
        self._ensure_directory()
        filepath = self.get_path(experience.experience_id)
        filepath.write_text(experience.to_json(indent=2), encoding="utf-8")
        return filepath

    def add(
        self,
        task: str,
        evaluation: EvaluationResult,
        reflection: ReflectionResult,
        trajectory_run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Experience:
        """Convenience helper to create, save, and return a new Experience."""
        exp = Experience(
            task=task,
            evaluation=evaluation,
            reflection=reflection,
            trajectory_run_id=trajectory_run_id,
            metadata=metadata or {},
        )
        self.save(exp)
        return exp

    def get(self, experience_id: str) -> Experience:
        """Retrieve an Experience by its ID. Raises FileNotFoundError if missing."""
        filepath = self.get_path(experience_id)
        if not filepath.is_file():
            raise FileNotFoundError(
                f"No experience found with ID '{experience_id}' at {filepath}"
            )
        return Experience.from_json(filepath.read_text(encoding="utf-8"))

    def exists(self, experience_id: str) -> bool:
        """Check if an experience with the given ID exists."""
        return self.get_path(experience_id).is_file()

    def list_experiences(self) -> List[Experience]:
        """Load and return all stored experiences."""
        if not self.storage_dir.is_dir():
            return []
        experiences = []
        for path in self.storage_dir.glob("*.json"):
            if path.is_file():
                try:
                    experiences.append(
                        Experience.from_json(path.read_text(encoding="utf-8"))
                    )
                except Exception:
                    pass
        return sorted(experiences, key=lambda e: e.created_at, reverse=True)

    def delete(self, experience_id: str) -> bool:
        """Delete a stored experience by ID if present."""
        filepath = self.get_path(experience_id)
        if filepath.is_file():
            filepath.unlink()
            return True
        return False

    def retrieve_relevant(
        self,
        query_task: str,
        limit: int = 3,
        min_score: float = 0.0,
        min_overlap: float = 0.0,
        only_successful: bool = False,
    ) -> List[Experience]:
        """Retrieve the most relevant past experiences for a new task prompt.

        Relevance is calculated using token overlap and keyword matching between
        the query task and stored experiences, combined with evaluation scores.
        """
        all_exp = self.list_experiences()
        if not all_exp:
            return []

        filtered = []
        for exp in all_exp:
            if exp.evaluation.score < min_score:
                continue
            if only_successful and not exp.evaluation.success:
                continue
            filtered.append(exp)

        if not filtered:
            return []

        query_tokens = _tokenize(query_task)
        if not query_tokens:
            return filtered[:limit]

        scored_experiences = []
        for exp in filtered:
            exp_text = f"{exp.task} {exp.reflection.what_worked} {exp.reflection.summary}"
            exp_tokens = _tokenize(exp_text)

            intersection = query_tokens.intersection(exp_tokens)
            overlap_score = (
                len(intersection) / len(query_tokens) if query_tokens else 0.0
            )

            if min_overlap > 0 and overlap_score < min_overlap:
                continue

            # Combine token overlap relevance (70%) with evaluation quality score (30%)
            total_relevance = (overlap_score * 0.7) + (exp.evaluation.score * 0.3)
            scored_experiences.append((total_relevance, exp))

        scored_experiences.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored_experiences[:limit]]


def _tokenize(text: str) -> Set[str]:
    """Helper to extract normalized word tokens from text."""
    words = re.findall(r"\w+", text.lower())
    stopwords = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "in",
        "of",
        "to",
        "is",
        "for",
        "with",
        "on",
        "this",
        "that",
    }
    return {w for w in words if len(w) > 1 and w not in stopwords}
