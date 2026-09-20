"""Tool abstraction for the terminal coding agent."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Tool(ABC):
    """Abstract base class for all agent tools."""

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}
    is_read_only: bool = False

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        is_read_only: Optional[bool] = None,
    ) -> None:
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if parameters is not None:
            self.parameters = parameters
        if is_read_only is not None:
            self.is_read_only = is_read_only

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with the provided arguments."""
        pass

    def to_function_schema(self) -> Dict[str, Any]:
        """Convert the tool into the function schema expected by model APIs."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
