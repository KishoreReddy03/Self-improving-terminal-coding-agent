"""Tests for tool abstraction and concrete tool implementations."""

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from coding_agent.tools import ReadFileTool, Tool


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


class TestReadFileTool(unittest.TestCase):
    """Test ReadFileTool implementation and error handling."""

    def setUp(self):
        self.tool = ReadFileTool()

    def test_tool_metadata(self):
        """Verify metadata of ReadFileTool."""
        self.assertEqual(self.tool.name, "read_file")
        self.assertTrue(self.tool.is_read_only)
        self.assertIn("path", self.tool.parameters["properties"])
        self.assertEqual(self.tool.parameters["required"], ["path"])

    def test_successful_file_read(self):
        """Verify reading a valid text file returns its exact contents."""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            tmp.write("Hello, World!\nSecond line.")
            tmp_path = tmp.name

        try:
            content = self.tool.execute(path=tmp_path)
            self.assertEqual(content, "Hello, World!\nSecond line.")
        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
