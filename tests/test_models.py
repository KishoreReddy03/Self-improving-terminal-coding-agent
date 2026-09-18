"""Tests for data models module."""

import unittest
from dataclasses import FrozenInstanceError

from coding_agent.models import ToolCall


class TestToolCall(unittest.TestCase):
    """Test ToolCall model initialization, serialization, and immutability."""

    def test_default_tool_call(self):
        """Verify default values for ToolCall fields."""
        tc = ToolCall(name="read_file")
        self.assertEqual(tc.name, "read_file")
        self.assertEqual(tc.arguments, {})
        self.assertIsNone(tc.id)

    def test_tool_call_with_arguments_and_id(self):
        """Verify custom values for ToolCall fields."""
        tc = ToolCall(
            name="run_command",
            arguments={"command": "python --version"},
            id="call_001",
        )
        self.assertEqual(tc.name, "run_command")
        self.assertEqual(tc.arguments, {"command": "python --version"})
        self.assertEqual(tc.id, "call_001")

    def test_to_dict_serialization(self):
        """Verify to_dict produces correct dictionary structure."""
        tc_without_id = ToolCall(name="list_dir", arguments={"path": "."})
        self.assertEqual(
            tc_without_id.to_dict(),
            {"name": "list_dir", "arguments": {"path": "."}},
        )

        tc_with_id = ToolCall(
            name="write_file", arguments={"path": "a.py"}, id="call_abc"
        )
        self.assertEqual(
            tc_with_id.to_dict(),
            {"name": "write_file", "arguments": {"path": "a.py"}, "id": "call_abc"},
        )

    def test_from_dict_deserialization(self):
        """Verify from_dict correctly parses dictionary into ToolCall object."""
        data = {
            "name": "edit_file",
            "arguments": {"path": "config.py", "content": "# comment"},
            "id": "call_xyz",
        }
        tc = ToolCall.from_dict(data)
        self.assertEqual(tc.name, "edit_file")
        self.assertEqual(tc.arguments, {"path": "config.py", "content": "# comment"})
        self.assertEqual(tc.id, "call_xyz")

    def test_from_dict_missing_name_raises_keyerror(self):
        """Verify missing 'name' key raises KeyError."""
        with self.assertRaises(KeyError):
            ToolCall.from_dict({"arguments": {}})

    def test_immutability(self):
        """Verify ToolCall fields cannot be reassigned due to frozen dataclass."""
        tc = ToolCall(name="read_file")
        with self.assertRaises(FrozenInstanceError):
            tc.name = "other_name"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
