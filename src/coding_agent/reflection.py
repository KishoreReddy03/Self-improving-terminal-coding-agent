"""Reflection component for analyzing completed agent run trajectories and evaluations.

This module provides ``ReflectionResult`` and ``ReflectionGenerator`` for generating
structured retrospectives on completed agent runs using a language model.
Reflection is kept strictly decoupled from execution and evaluation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from coding_agent.client import LLMClient
from coding_agent.evaluator import EvaluationResult
from coding_agent.models import ModelResponse
from coding_agent.run_record import RunEvent, RunEventType, RunRecord


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


REFLECTION_SYSTEM_PROMPT = (
    "You are an expert agent performance analyst. Your job is to analyze completed "
    "agent execution trajectories along with their evaluation results, and output "
    "a concise, structured JSON retrospective reflection.\n\n"
    "You MUST respond ONLY with a valid JSON object matching the schema below:\n"
    "{\n"
    '  "what_worked": "<Concise summary of successful actions, tools, or decisions>",\n'
    '  "what_failed": "<Summary of errors, failed tool calls, or limits hit>",\n'
    '  "why_it_failed": "<Root-cause analysis of why errors or failures occurred>",\n'
    '  "what_to_do_differently": "<Actionable recommendations for future runs>",\n'
    '  "summary": "<Overall high-level retrospective summary>"\n'
    "}"
)


from coding_agent.client import LLMClient
from coding_agent.evaluator import EvaluationResult
from coding_agent.run_record import RunRecord


class ReflectionGenerator:
    """Generates structured reflections for completed runs using an LLM model client."""

    def __init__(self, client: Optional[LLMClient] = None) -> None:
        self.client = client or LLMClient()

    def _format_reflection_prompt(
        self, record: RunRecord, evaluation: EvaluationResult
    ) -> str:
        """Build the user prompt containing trajectory details and evaluation data."""
        tool_summary_lines = []
        for e in record.events:
            if e.event_type == RunEventType.TOOL_CALL:
                t_name = e.data.get("tool_name", "")
                t_args = e.data.get("arguments", {})
                tool_summary_lines.append(f"- Tool Called: {t_name}({json.dumps(t_args)})")
            elif e.event_type == RunEventType.TOOL_RESULT:
                t_name = e.data.get("tool_name", "")
                is_err = e.data.get("is_error", False)
                out_snippet = str(e.data.get("output", ""))[:200]
                status = "ERROR" if is_err else "SUCCESS"
                tool_summary_lines.append(f"  Result [{status}]: {out_snippet}")

        tool_summary_str = (
            "\n".join(tool_summary_lines) if tool_summary_lines else "None"
        )

        prompt = (
            f"Please analyze the following agent execution trajectory and evaluation:\n\n"
            f"=== TASK ===\n{record.task}\n\n"
            f"=== RUN SUMMARY ===\n"
            f"- Outcome: {record.outcome.value if record.outcome else 'unknown'}\n"
            f"- Steps Taken: {record.steps_taken}\n"
            f"- Final Response: {record.final_response or 'None'}\n"
            f"- Error Message: {record.error_message or 'None'}\n\n"
            f"=== TOOL USAGE HISTORY ===\n{tool_summary_str}\n\n"
            f"=== EVALUATION RESULT ===\n"
            f"- Success: {evaluation.success}\n"
            f"- Score: {evaluation.score}\n"
            f"- Reason: {evaluation.reason}\n"
            f"- Details: {json.dumps(evaluation.details, indent=2)}\n\n"
            f"Produce your structured JSON reflection now."
        )
        return prompt
