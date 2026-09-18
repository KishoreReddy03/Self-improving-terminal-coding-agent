"""Data models for the terminal coding agent."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ToolCall:
    """Represents a tool call requested by the model or agent."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool call to dictionary representation."""
        result: Dict[str, Any] = {
            "name": self.name,
            "arguments": dict(self.arguments),
        }
        if self.id is not None:
            result["id"] = self.id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """Construct a ToolCall instance from dictionary representation."""
        if "name" not in data:
            raise KeyError("ToolCall dictionary must contain 'name' key.")
        return cls(
            name=data["name"],
            arguments=dict(data.get("arguments") or {}),
            id=data.get("id"),
        )
