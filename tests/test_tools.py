"""Tests for tool abstraction and concrete tool implementations."""

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from coding_agent.tools import EditFileTool, ReadFileTool, Tool, WriteFileTool


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

    def test_missing_path_parameter(self):
        """Verify error message when path argument is empty."""
        result = self.tool.execute(path="")
        self.assertIn("Error: 'path' parameter is required", result)

    def test_nonexistent_file(self):
        """Verify error message when file does not exist."""
        result = self.tool.execute(path="non_existent_file_12345.txt")
        self.assertIn("Error: File 'non_existent_file_12345.txt' does not exist", result)

    def test_directory_path_failure(self):
        """Verify error message when path points to a directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = self.tool.execute(path=tmp_dir)
            self.assertIn("is a directory, not a file", result)

    def test_unicode_decode_error(self):
        """Verify error message when reading non-UTF-8 binary content."""
        with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
            tmp.write(bytes([0x80, 0x81, 0xFE, 0xFF]))
            tmp_path = tmp.name

        try:
            result = self.tool.execute(path=tmp_path)
            self.assertIn("Unable to decode file", result)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestWriteFileTool(unittest.TestCase):
    """Test WriteFileTool implementation and error handling."""

    def setUp(self):
        self.tool = WriteFileTool()

    def test_tool_metadata(self):
        """Verify metadata of WriteFileTool."""
        self.assertEqual(self.tool.name, "write_file")
        self.assertFalse(self.tool.is_read_only)
        self.assertEqual(self.tool.parameters["required"], ["path", "content"])

    def test_create_new_file_with_parent_directories(self):
        """Verify writing creates new file and required parent directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "nested" / "folder" / "test.txt"
            result = self.tool.execute(path=str(file_path), content="Hello, Nested World!")

            self.assertIn("Successfully wrote", result)
            self.assertTrue(file_path.exists())
            self.assertEqual(file_path.read_text(encoding="utf-8"), "Hello, Nested World!")

    def test_overwrite_existing_file(self):
        """Verify writing overwrites existing file content."""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            tmp.write("Initial content")
            tmp_path = tmp.name

        try:
            result = self.tool.execute(path=tmp_path, content="Overwritten content")
            self.assertIn("Successfully wrote", result)
            self.assertEqual(Path(tmp_path).read_text(encoding="utf-8"), "Overwritten content")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_missing_path_parameter(self):
        """Verify error message when path argument is empty."""
        result = self.tool.execute(path="", content="some content")
        self.assertIn("Error: 'path' parameter is required", result)

    def test_directory_path_failure(self):
        """Verify error message when target path is a directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = self.tool.execute(path=tmp_dir, content="content")
            self.assertIn("is a directory, not a file", result)


class TestEditFileTool(unittest.TestCase):
    """Test EditFileTool implementation and exact string replacement."""

    def setUp(self):
        self.tool = EditFileTool()

    def test_tool_metadata(self):
        """Verify metadata of EditFileTool."""
        self.assertEqual(self.tool.name, "edit_file")
        self.assertFalse(self.tool.is_read_only)
        self.assertEqual(self.tool.parameters["required"], ["path", "old_str", "new_str"])

    def test_successful_string_replacement(self):
        """Verify exact string replacement updates file correctly."""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            tmp.write("foo bar baz")
            tmp_path = tmp.name

        try:
            result = self.tool.execute(path=tmp_path, old_str="bar", new_str="qux")
            self.assertIn("Successfully edited", result)
            self.assertEqual(Path(tmp_path).read_text(encoding="utf-8"), "foo qux baz")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_target_text_not_found(self):
        """Verify clear error message when target string is not present in file."""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            tmp.write("foo bar baz")
            tmp_path = tmp.name

        try:
            result = self.tool.execute(path=tmp_path, old_str="non_existent", new_str="replacement")
            self.assertIn("Target text to replace was not found", result)
            self.assertEqual(Path(tmp_path).read_text(encoding="utf-8"), "foo bar baz")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_missing_old_str_parameter(self):
        """Verify error message when old_str argument is empty."""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            tmp.write("content")
            tmp_path = tmp.name

        try:
            result = self.tool.execute(path=tmp_path, old_str="", new_str="new")
            self.assertIn("Error: 'old_str' parameter is required", result)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_nonexistent_file(self):
        """Verify error message when file to edit does not exist."""
        result = self.tool.execute(path="missing_file_xyz.txt", old_str="a", new_str="b")
        self.assertIn("Error: File 'missing_file_xyz.txt' does not exist", result)


if __name__ == "__main__":
    unittest.main()
