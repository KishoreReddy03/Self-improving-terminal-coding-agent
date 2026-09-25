"""Tests for tool call parser and stream accumulator module."""

import json
import unittest

from coding_agent.models import ModelResponse, ToolCall
from coding_agent.parser import ResponseParser, StreamToolCallAccumulator


class TestResponseParser(unittest.TestCase):
    """Test ResponseParser for parsing complete provider response objects."""

    def test_parse_normal_text_response(self):
        """Verify parsing a normal text assistant response with no tool calls."""
        raw_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello! I am an AI coding assistant.",
                    }
                }
            ]
        }
        response = ResponseParser.parse(raw_response)

        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.content, "Hello! I am an AI coding assistant.")
        self.assertFalse(response.has_tool_calls())
        self.assertEqual(response.tool_calls, [])

    def test_parse_single_tool_call_with_json_string_args(self):
        """Verify parsing a single tool call with arguments as a JSON string."""
        raw_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I will read the requested file.",
                        "tool_calls": [
                            {
                                "id": "call_12345",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "src/main.py"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        response = ResponseParser.parse(raw_response)

        self.assertEqual(response.content, "I will read the requested file.")
        self.assertTrue(response.has_tool_calls())
        self.assertEqual(len(response.tool_calls), 1)

        tool_call = response.tool_calls[0]
        self.assertEqual(tool_call.name, "read_file")
        self.assertEqual(tool_call.arguments, {"path": "src/main.py"})
        self.assertEqual(tool_call.id, "call_12345")

    def test_parse_tool_call_with_dict_arguments(self):
        """Verify parsing a tool call when arguments are provided directly as a dictionary."""
        raw_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_dict",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": {"path": "out.txt", "content": "data"},
                                },
                            }
                        ],
                    }
                }
            ]
        }
        response = ResponseParser.parse(raw_response)

        self.assertIsNone(response.content)
        self.assertTrue(response.has_tool_calls())
        self.assertEqual(response.tool_calls[0].arguments, {"path": "out.txt", "content": "data"})

    def test_parse_multiple_tool_calls(self):
        """Verify parsing multiple tool calls in a single assistant message."""
        raw_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Executing two file operations.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {"name": "edit_file", "arguments": '{"path": "b.txt", "old_str": "x", "new_str": "y"}'},
                            },
                        ],
                    }
                }
            ]
        }
        response = ResponseParser.parse(raw_response)

        self.assertEqual(len(response.tool_calls), 2)
        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(response.tool_calls[1].name, "edit_file")

    def test_parse_invalid_json_arguments_fallback(self):
        """Verify invalid JSON arguments fallback to raw_arguments key."""
        raw_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "function": {"name": "run_cmd", "arguments": "{invalid_json"},
                            }
                        ],
                    }
                }
            ]
        }
        response = ResponseParser.parse(raw_response)
        self.assertEqual(response.tool_calls[0].arguments, {"raw_arguments": "{invalid_json"})


if __name__ == "__main__":
    unittest.main()
