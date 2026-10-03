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

    def save(self, record: RunRecord) -> Path:
        """Persist a RunRecord to disk as structured JSON.

        Parameters
        ----------
        record:
            The RunRecord instance to persist.

        Returns
        -------
        Path:
            The file path where the trajectory was written.
        """
        self._ensure_directory()
        filepath = self.get_path(record.run_id)
        json_data = record.to_json(indent=2)
        filepath.write_text(json_data, encoding="utf-8")
        return filepath

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
