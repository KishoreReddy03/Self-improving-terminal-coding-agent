"""Data models for the terminal coding agent."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
class ModelResponse:
    """Represents a structured response returned by the LLM client layer."""

    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw_response: Optional[Dict[str, Any]] = None

    def has_tool_calls(self) -> bool:
        """Check if the model response contains tool call requests."""
        return len(self.tool_calls) > 0


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


@dataclass(frozen=True)
class AgentResult:
    """Result returned by running the core agent loop."""

    final_response: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    steps_taken: int = 0
    completed: bool = True


