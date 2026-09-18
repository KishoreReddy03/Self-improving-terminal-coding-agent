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


@dataclass(frozen=True)
class AgentConfig:
    """Runtime configuration settings for the agent."""

    api_key: str = ""
    provider_url: str = "poolside/laguna-s-2.1:free"
    model_name: str = "poolside/laguna-s-2.1:free"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 60.0
    max_steps: int = 40
    system_prompt: Optional[str] = None
