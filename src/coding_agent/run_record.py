"""Structured representation of one complete agent execution run.

This module provides immutable, serialisable data-classes that capture every
meaningful event that occurs during a single agent run.  The structures are
intentionally kept separate from long-term memory and reflection logic – they
exist solely so that any run can be represented, stored, compared, or
inspected as plain data.

Typical event sequence for a successful tool-using run
-------------------------------------------------------
TASK_START → (LLM_CALL → TOOL_CALL → TOOL_RESULT)* → LLM_CALL → FINAL_RESPONSE
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Outcome enumeration
# ---------------------------------------------------------------------------


class RunOutcome(str, Enum):
    """High-level result of a complete agent run."""

    SUCCESS = "success"
    """The model produced a final text response within the step limit."""

    MAX_STEPS_REACHED = "max_steps_reached"
    """The agent exhausted its step budget without a final text response."""

    ERROR = "error"
    """An unhandled exception terminated the run prematurely."""


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class RunEventType(str, Enum):
    """Discriminator for the kind of event stored in a RunRecord."""

    TASK_START = "task_start"
    """Emitted once at the start of a run with the initial task description."""

    LLM_CALL = "llm_call"
    """Emitted each time the agent sends messages to the model."""

    TOOL_CALL = "tool_call"
    """Emitted when the agent dispatches a tool (includes approval decision)."""

    TOOL_RESULT = "tool_result"
    """Emitted after a tool finishes (or is denied/errored)."""

    FINAL_RESPONSE = "final_response"
    """Emitted when the model produces a conclusive text answer."""

    ERROR_EVENT = "error"
    """Emitted when an exception propagates out of the agent loop."""


# ---------------------------------------------------------------------------
# Individual event dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunEvent:
    """A single timestamped event that occurred during an agent run.

    Attributes
    ----------
    event_type:
        Discriminator describing what happened at this point in the run.
    step:
        The loop-iteration index (1-based) in which this event occurred.
        ``0`` is reserved for run-level events (TASK_START, FINAL_RESPONSE,
        ERROR_EVENT) that are not tied to a specific loop step.
    data:
        Arbitrary payload for the event.  Structure depends on *event_type*:

        - ``TASK_START``   → ``{"task": str}``
        - ``LLM_CALL``     → ``{"message_count": int, "has_tools": bool}``
        - ``TOOL_CALL``    → ``{"tool_name": str, "arguments": dict,
                                "tool_call_id": str, "approved": bool | None}``
          (*approved* is ``None`` for read-only tools that skip approval.)
        - ``TOOL_RESULT``  → ``{"tool_name": str, "tool_call_id": str,
                                "output": str, "is_error": bool}``
        - ``FINAL_RESPONSE`` → ``{"content": str}``
        - ``ERROR_EVENT``  → ``{"error_type": str, "message": str}``
    timestamp:
        UTC timestamp of when this event was recorded.  Auto-set on creation.
    """

    event_type: RunEventType
    step: int
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this event to a plain dictionary (JSON-safe)."""
        return {
            "event_type": self.event_type.value,
            "step": self.step,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunEvent":
        """Reconstruct a RunEvent from its dictionary representation."""
        return cls(
            event_type=RunEventType(data["event_type"]),
            step=data["step"],
            data=dict(data.get("data") or {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


# ---------------------------------------------------------------------------
# Run record (top-level container)
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """Complete, structured record of one agent execution run.

    A ``RunRecord`` is built incrementally by the agent loop and becomes
    immutable (logically) once ``outcome`` is set.  It is intentionally
    *separate* from any persistence or memory layer so it can be used,
    tested, and reasoned about as pure data.

    Attributes
    ----------
    run_id:
        Unique identifier for this run.  Auto-generated as a UUID4 string.
    task:
        The initial user task or prompt that triggered this run.
    outcome:
        High-level result once the run has finished.  ``None`` while the run
        is still in progress.
    final_response:
        The model's last text answer, or ``None`` if the run did not complete
        successfully.
    steps_taken:
        Total number of loop iterations executed.
    events:
        Ordered list of every :class:`RunEvent` recorded during the run.
    messages:
        Full conversation history (same format as the agent's internal
        message list) as it stood at the end of the run.
    error_message:
        Exception message if *outcome* is ``RunOutcome.ERROR``, else ``None``.
    started_at:
        UTC timestamp when the run started.
    finished_at:
        UTC timestamp when the run finished.  ``None`` while still running.
    metadata:
        Arbitrary key/value pairs callers may attach (e.g. model name,
        temperature, run tags).  Not used by the core agent loop itself.
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task: str = ""
    outcome: Optional[RunOutcome] = None
    final_response: Optional[str] = None
    steps_taken: int = 0
    events: List[RunEvent] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers (all read-only views – they do not mutate state)
    # ------------------------------------------------------------------

    @property
    def duration_seconds(self) -> Optional[float]:
        """Wall-clock seconds the run took, or ``None`` if not yet finished."""
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def tool_calls(self) -> List[RunEvent]:
        """Return all TOOL_CALL events in chronological order."""
        return [e for e in self.events if e.event_type == RunEventType.TOOL_CALL]

    @property
    def tool_results(self) -> List[RunEvent]:
        """Return all TOOL_RESULT events in chronological order."""
        return [e for e in self.events if e.event_type == RunEventType.TOOL_RESULT]

    @property
    def error_events(self) -> List[RunEvent]:
        """Return all ERROR_EVENT events in chronological order."""
        return [e for e in self.events if e.event_type == RunEventType.ERROR_EVENT]

    @property
    def llm_calls(self) -> List[RunEvent]:
        """Return all LLM_CALL events in chronological order."""
        return [e for e in self.events if e.event_type == RunEventType.LLM_CALL]

    def was_successful(self) -> bool:
        """Return True if the run completed with a final text response."""
        return self.outcome == RunOutcome.SUCCESS

    def tool_call_count(self) -> int:
        """Return total number of tool calls dispatched during the run."""
        return len(self.tool_calls)

    def error_tool_results(self) -> List[RunEvent]:
        """Return TOOL_RESULT events that carry error output."""
        return [e for e in self.tool_results if e.data.get("is_error", False)]

    # ------------------------------------------------------------------
    # Event recording helpers (called by the agent loop)
    # ------------------------------------------------------------------

    def add_event(self, event: RunEvent) -> None:
        """Append a new event to the event log."""
        self.events.append(event)

    def record_task_start(self, task: str) -> None:
        """Record the TASK_START event."""
        self.task = task
        self.add_event(
            RunEvent(
                event_type=RunEventType.TASK_START,
                step=0,
                data={"task": task},
            )
        )

    def record_llm_call(self, step: int, message_count: int, has_tools: bool) -> None:
        """Record that the agent sent a request to the model."""
        self.add_event(
            RunEvent(
                event_type=RunEventType.LLM_CALL,
                step=step,
                data={"message_count": message_count, "has_tools": has_tools},
            )
        )

    def record_tool_call(
        self,
        step: int,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_call_id: str,
        approved: Optional[bool],
    ) -> None:
        """Record a tool dispatch attempt (before execution)."""
        self.add_event(
            RunEvent(
                event_type=RunEventType.TOOL_CALL,
                step=step,
                data={
                    "tool_name": tool_name,
                    "arguments": dict(arguments),
                    "tool_call_id": tool_call_id,
                    "approved": approved,
                },
            )
        )

    def record_tool_result(
        self,
        step: int,
        tool_name: str,
        tool_call_id: str,
        output: str,
        is_error: bool,
    ) -> None:
        """Record the output (or error) of a tool execution."""
        self.add_event(
            RunEvent(
                event_type=RunEventType.TOOL_RESULT,
                step=step,
                data={
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "output": output,
                    "is_error": is_error,
                },
            )
        )

    def record_final_response(self, content: Optional[str]) -> None:
        """Record the model's conclusive text answer."""
        self.add_event(
            RunEvent(
                event_type=RunEventType.FINAL_RESPONSE,
                step=0,
                data={"content": content or ""},
            )
        )

    def record_error(self, error_type: str, message: str) -> None:
        """Record an unhandled exception that terminated the run."""
        self.add_event(
            RunEvent(
                event_type=RunEventType.ERROR_EVENT,
                step=0,
                data={"error_type": error_type, "message": message},
            )
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full RunRecord to a plain, JSON-safe dictionary."""
        return {
            "run_id": self.run_id,
            "task": self.task,
            "outcome": self.outcome.value if self.outcome is not None else None,
            "final_response": self.final_response,
            "steps_taken": self.steps_taken,
            "events": [e.to_dict() for e in self.events],
            "messages": list(self.messages),
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "metadata": dict(self.metadata),
        }

    def to_json(self, **kwargs: Any) -> str:
        """Return the RunRecord serialised as a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRecord":
        """Reconstruct a RunRecord from its dictionary representation."""
        record = cls(
            run_id=data["run_id"],
            task=data.get("task", ""),
            outcome=RunOutcome(data["outcome"]) if data.get("outcome") else None,
            final_response=data.get("final_response"),
            steps_taken=data.get("steps_taken", 0),
            events=[RunEvent.from_dict(e) for e in data.get("events", [])],
            messages=list(data.get("messages", [])),
            error_message=data.get("error_message"),
            started_at=datetime.fromisoformat(data["started_at"]),
            finished_at=(
                datetime.fromisoformat(data["finished_at"])
                if data.get("finished_at")
                else None
            ),
            metadata=dict(data.get("metadata", {})),
        )
        return record

    @classmethod
    def from_json(cls, text: str) -> "RunRecord":
        """Reconstruct a RunRecord from a JSON string."""
        return cls.from_dict(json.loads(text))
