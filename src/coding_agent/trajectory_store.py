"""Local persistence and retrieval of agent execution trajectories.

This module provides ``TrajectoryStore`` for saving and loading completed
``RunRecord`` trajectories as local JSON files. Recording remains decoupled from
the core agent execution loop: the agent populates an in-memory ``RunRecord``,
and callers (CLI, APIs, test hooks) decide when and where to persist it using
``TrajectoryStore``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

from coding_agent.evaluator import EvaluationResult
from coding_agent.reflection import ReflectionResult
from coding_agent.run_record import RunRecord


class TrajectoryStore:
    """Persists and retrieves completed agent run trajectories as structured JSON.

    Each trajectory is stored as an individual JSON file named ``{run_id}.json``
    under the target storage directory.

    Parameters
    ----------
    storage_dir:
        Directory path where JSON trajectories will be saved and loaded.
        Defaults to ``~/.coding_agent/trajectories``.
    """

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None) -> None:
        if storage_dir is None:
            self.storage_dir = Path.home() / ".coding_agent" / "trajectories"
        else:
            self.storage_dir = Path(storage_dir)

    def _ensure_directory(self) -> None:
        """Create storage directory if it does not exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, run_id: str) -> Path:
        """Return the file path for a given run_id."""
        return self.storage_dir / f"{run_id}.json"

    def save(
        self,
        record: RunRecord,
        evaluation: Optional[EvaluationResult] = None,
        reflection: Optional[ReflectionResult] = None,
    ) -> Path:
        """Persist a RunRecord to disk as structured JSON.

        Parameters
        ----------
        record:
            The RunRecord instance to persist.
        evaluation:
            Optional EvaluationResult to attach to the record's metadata.
        reflection:
            Optional ReflectionResult to attach to the record's metadata.

        Returns
        -------
        Path:
            The file path where the trajectory was written.
        """
        self._ensure_directory()
        if evaluation is not None:
            record.metadata["evaluation"] = evaluation.to_dict()
        if reflection is not None:
            record.metadata["reflection"] = reflection.to_dict()

        filepath = self.get_path(record.run_id)
        json_data = record.to_json(indent=2)
        filepath.write_text(json_data, encoding="utf-8")
        return filepath

    def save_run_analysis(
        self,
        record: RunRecord,
        evaluation: Optional[EvaluationResult] = None,
        reflection: Optional[ReflectionResult] = None,
    ) -> Path:
        """Convenience wrapper for saving a run record along with its analysis."""
        return self.save(record, evaluation=evaluation, reflection=reflection)

    def get_evaluation(self, run_id: str) -> Optional[EvaluationResult]:
        """Retrieve stored EvaluationResult for a run_id if present."""
        record = self.load(run_id)
        eval_dict = record.metadata.get("evaluation")
        if eval_dict and isinstance(eval_dict, dict):
            return EvaluationResult.from_dict(eval_dict)
        return None

    def get_reflection(self, run_id: str) -> Optional[ReflectionResult]:
        """Retrieve stored ReflectionResult for a run_id if present."""
        record = self.load(run_id)
        refl_dict = record.metadata.get("reflection")
        if refl_dict and isinstance(refl_dict, dict):
            return ReflectionResult.from_dict(refl_dict)
        return None

    def load(self, run_id: str) -> RunRecord:
        """Load and deserialise a RunRecord by its run_id.

        Raises
        ------
        FileNotFoundError:
            If no trajectory file exists for the specified run_id.
        """
        filepath = self.get_path(run_id)
        if not filepath.is_file():
            raise FileNotFoundError(
                f"No trajectory found for run_id '{run_id}' at {filepath}"
            )
        content = filepath.read_text(encoding="utf-8")
        return RunRecord.from_json(content)

    def exists(self, run_id: str) -> bool:
        """Check whether a trajectory file exists for the given run_id."""
        return self.get_path(run_id).is_file()

    def list_run_ids(self) -> List[str]:
        """List all stored run IDs in sorted order."""
        if not self.storage_dir.is_dir():
            return []
        return sorted([f.stem for f in self.storage_dir.glob("*.json") if f.is_file()])

    def list_records(self) -> List[RunRecord]:
        """Load and return all valid RunRecords stored in the directory."""
        records: List[RunRecord] = []
        for run_id in self.list_run_ids():
            try:
                records.append(self.load(run_id))
            except Exception:
                pass
        return records

    def count(self) -> int:
        """Return total number of stored trajectories."""
        return len(self.list_run_ids())

    def list_records_since(self, min_timestamp: float) -> List[RunRecord]:
        """Return all valid RunRecords created at or after min_timestamp (POSIX time)."""
        records = self.list_records()
        return [r for r in records if r.started_at.timestamp() >= min_timestamp]


    def delete(self, run_id: str) -> bool:
        """Delete a stored trajectory file by run_id if it exists.

        Returns True if deleted, False if file did not exist.
        """
        filepath = self.get_path(run_id)
        if filepath.is_file():
            filepath.unlink()
            return True
        return False

