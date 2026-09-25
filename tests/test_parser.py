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


class TestStreamToolCallAccumulator(unittest.TestCase):
    """Test StreamToolCallAccumulator for collecting and parsing streamed delta chunks."""

    def setUp(self):
        self.accumulator = StreamToolCallAccumulator()

    def test_stream_normal_text_fragments(self):
        """Verify accumulating text fragments over multiple chunks."""
        chunks = [
            {"choices": [{"delta": {"role": "assistant", "content": "Hello "}}]},
            {"choices": [{"delta": {"content": "world!"}}]},
        ]
        for chunk in chunks:
            self.accumulator.add_chunk(chunk)

        response = self.accumulator.to_model_response()
        self.assertEqual(response.content, "Hello world!")
        self.assertFalse(response.has_tool_calls())

    def test_stream_fragmented_tool_call(self):
        """Verify accumulating a tool call whose name and JSON arguments are split across chunks."""
        chunks = [
            # Chunk 1: Role, ID, initial function name fragment
            {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": "Reading file now.",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "stream_call_001",
                                    "type": "function",
                                    "function": {"name": "read_"},
                                }
                            ],
                        }
                    }
                ]
            },
            # Chunk 2: Function name completion & argument prefix
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"name": "file", "arguments": '{"path":'},
                                }
                            ]
                        }
                    }
                ]
            },
            # Chunk 3: Argument completion
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ' "src/app.py"}'},
                                }
                            ]
                        }
                    }
                ]
            },
        ]

        for chunk in chunks:
            self.accumulator.add_chunk(chunk)

        response = self.accumulator.to_model_response()
        self.assertEqual(response.content, "Reading file now.")
        self.assertTrue(response.has_tool_calls())
        self.assertEqual(len(response.tool_calls), 1)

        tool_call = response.tool_calls[0]
        self.assertEqual(tool_call.id, "stream_call_001")
        self.assertEqual(tool_call.name, "read_file")
        self.assertEqual(tool_call.arguments, {"path": "src/app.py"})

    def test_stream_multiple_concurrent_tool_calls(self):
        """Verify accumulating multiple tool calls with distinct delta index values."""
        chunks = [
            # Tool call 0
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "tc_0",
                                    "function": {"name": "read_file", "arguments": '{"path": "file1.txt"}'},
                                }
                            ]
                        }
                    }
                ]
            },
            # Tool call 1
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "tc_1",
                                    "function": {"name": "write_file", "arguments": '{"path": "file2.txt", "content": "hi"}'},
                                }
                            ]
                        }
                    }
                ]
            },
        ]

        for chunk in chunks:
            self.accumulator.add_chunk(chunk)

        response = self.accumulator.to_model_response()
        self.assertEqual(len(response.tool_calls), 2)
        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(response.tool_calls[0].arguments, {"path": "file1.txt"})
        self.assertEqual(response.tool_calls[1].name, "write_file")
        self.assertEqual(response.tool_calls[1].arguments, {"path": "file2.txt", "content": "hi"})

    def test_stream_accumulator_reset(self):
        """Verify reset clears all accumulated state."""
        self.accumulator.add_chunk({"choices": [{"delta": {"content": "Some text"}}]})
        self.accumulator.reset()

        response = self.accumulator.to_model_response()
        self.assertIsNone(response.content)
        self.assertEqual(response.tool_calls, [])

    def test_stream_json_string_chunks(self):
        """Verify accumulator accepts JSON formatted string chunks directly."""
        json_chunk = json.dumps({
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "json_str_call",
                                "function": {"name": "shell_command", "arguments": '{"command": "dir"}'},
                            }
                        ]
                    }
                }
            ]
        })
        self.accumulator.add_chunk(json_chunk)
        response = self.accumulator.to_model_response()

        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].name, "shell_command")
        self.assertEqual(response.tool_calls[0].arguments, {"command": "dir"})


if __name__ == "__main__":
    unittest.main()

