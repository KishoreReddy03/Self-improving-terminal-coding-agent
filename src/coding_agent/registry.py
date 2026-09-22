"""Central tool registry for discovering and managing agent tools."""

from typing import Any, Dict, List, Optional

from coding_agent.tools import EditFileTool, ReadFileTool, Tool, WriteFileTool


class ToolRegistry:
    """Registry for managing and discovering agent tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance."""
        if not tool.name:
            raise ValueError("Cannot register a tool with an empty name.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Retrieve a registered tool by its name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """Return a list of all registered tool instances."""
        return list(self._tools.values())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return function declaration schemas for all registered tools."""
        return [tool.to_function_schema() for tool in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        """Check if a tool name is registered."""
        return name in self._tools

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    @classmethod
    def create_default(cls) -> "ToolRegistry":
        """Construct a ToolRegistry instance pre-populated with standard built-in tools."""
        registry = cls()
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())
        registry.register(EditFileTool())
        return registry
