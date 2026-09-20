"""Tests for tool abstraction module."""

import unittest
from typing import Any, Dict

from coding_agent.tools import Tool


class DummyTool(Tool):
    """Concrete test implementation of Tool base class."""

    name = "dummy_tool"
    description = "A dummy tool for testing purposes."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    }
    is_read_only = True

    def execute(self, query: str = "") -> str:
        return f"Processed: {query}"


class TestToolAbstraction(unittest.TestCase):
    """Test Tool base class functionality and schema generation."""

    def test_cannot_instantiate_abstract_tool(self):
        """Ensure direct instantiation of Tool raises TypeError due to abstract execute method."""
        with self.assertRaises(TypeError):
            Tool()  # type: ignore[abstract]

    def test_subclass_properties(self):
        """Verify subclass inherits and exposes tool metadata correctly."""
        tool = DummyTool()
        self.assertEqual(tool.name, "dummy_tool")
        self.assertEqual(tool.description, "A dummy tool for testing purposes.")
        self.assertTrue(tool.is_read_only)
        self.assertEqual(tool.parameters["type"], "object")

    def test_override_properties_in_init(self):
        """Verify properties can be overridden during initialization."""
        tool = DummyTool(
            name="custom_dummy",
            description="Custom description",
            is_read_only=False,
        )
        self.assertEqual(tool.name, "custom_dummy")
        self.assertEqual(tool.description, "Custom description")
        self.assertFalse(tool.is_read_only)

    def test_execute_method(self):
        """Verify execute method works as expected on concrete subclass."""
        tool = DummyTool()
        result = tool.execute(query="hello")
        self.assertEqual(result, "Processed: hello")

    def test_to_function_schema(self):
        """Verify to_function_schema converts tool into model API format."""
        tool = DummyTool()
        schema = tool.to_function_schema()
        expected = {
            "type": "function",
            "function": {
                "name": "dummy_tool",
                "description": "A dummy tool for testing purposes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        }
        self.assertEqual(schema, expected)


if __name__ == "__main__":
    unittest.main()
