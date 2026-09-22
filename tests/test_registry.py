"""Tests for central tool registry module."""

import unittest
from typing import Any

from coding_agent.registry import ToolRegistry
from coding_agent.tools import Tool


class MockTool(Tool):
    """Mock tool implementation for registry testing."""

    name = "mock_tool"
    description = "Mock tool description."
    parameters = {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> str:
        return "mock result"


class TestToolRegistry(unittest.TestCase):
    """Test ToolRegistry functionality and default tool registration."""

    def test_empty_registry(self):
        """Verify new ToolRegistry is initially empty."""
        registry = ToolRegistry()
        self.assertEqual(len(registry), 0)
        self.assertEqual(registry.list_tools(), [])
        self.assertEqual(registry.get_schemas(), [])
        self.assertIsNone(registry.get("non_existent"))

    def test_register_and_get_tool(self):
        """Verify registering a tool allows retrieval by name."""
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)

        self.assertEqual(len(registry), 1)
        self.assertIn("mock_tool", registry)
        self.assertIs(registry.get("mock_tool"), tool)

    def test_register_tool_without_name_raises_valueerror(self):
        """Verify registering a tool with an empty name raises ValueError."""
        registry = ToolRegistry()
        invalid_tool = MockTool(name="")
        with self.assertRaises(ValueError):
            registry.register(invalid_tool)

    def test_get_schemas(self):
        """Verify get_schemas returns function schemas for registered tools."""
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)

        schemas = registry.get_schemas()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["type"], "function")
        self.assertEqual(schemas[0]["function"]["name"], "mock_tool")


if __name__ == "__main__":
    unittest.main()
