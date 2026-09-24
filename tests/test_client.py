"""Tests for LLM client layer module."""

import json
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from coding_agent.client import LLMClient
from coding_agent.models import AgentConfig, ModelResponse, ToolCall


class TestLLMClient(unittest.TestCase):
    """Test LLMClient initialization, request payload generation, and response parsing."""

    def test_client_initialization_defaults(self):
        """Verify client initializes with defaults from AgentConfig."""
        config = AgentConfig(
            api_key="test-key",
            provider_url="https://api.example.com/v1",
            model_name="test-model",
        )
        client = LLMClient(config=config)

        self.assertEqual(client.api_key, "test-key")
        self.assertEqual(client.provider_url, "https://api.example.com/v1")
        self.assertEqual(client.model_name, "test-model")
        self.assertEqual(client.max_tokens, 4096)
        self.assertEqual(client.temperature, 0.7)
        self.assertEqual(client.timeout, 60.0)

    def test_client_parameter_overrides(self):
        """Verify client parameters can be explicitly overridden in constructor."""
        config = AgentConfig(api_key="default-key", model_name="default-model")
        client = LLMClient(
            config=config,
            api_key="override-key",
            model_name="override-model",
            provider_url="https://custom.provider.com/v1",
        )

        self.assertEqual(client.api_key, "override-key")
        self.assertEqual(client.model_name, "override-model")
        self.assertEqual(client.provider_url, "https://custom.provider.com/v1")

    def test_generate_text_response(self):
        """Verify generating a simple text response without tool calls."""
        captured_payload = {}
        captured_headers = {}

        def mock_transport(payload, headers, timeout):
            nonlocal captured_payload, captured_headers
            captured_payload = payload
            captured_headers = headers
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello! How can I help you?",
                        }
                    }
                ]
            }

        client = LLMClient(
            api_key="sk-test-123",
            model_name="gpt-4o-mini",
            transport=mock_transport,
        )

        messages = [{"role": "user", "content": "Hi"}]
        response = client.generate_response(messages)

        self.assertEqual(captured_payload["model"], "gpt-4o-mini")
        self.assertEqual(captured_payload["messages"], messages)
        self.assertEqual(captured_headers["Authorization"], "Bearer sk-test-123")
        self.assertEqual(captured_headers["Content-Type"], "application/json")
        self.assertNotIn("tools", captured_payload)

        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.content, "Hello! How can I help you?")
        self.assertFalse(response.has_tool_calls())
        self.assertEqual(response.tool_calls, [])

    def test_generate_response_with_tool_calls(self):
        """Verify client handles function tool definitions and parses returned tool calls."""
        tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file content",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

        captured_payload = {}

        def mock_transport(payload, headers, timeout):
            nonlocal captured_payload
            captured_payload = payload
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I will read the file for you.",
                            "tool_calls": [
                                {
                                    "id": "call_987",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path": "config.json"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        client = LLMClient(model_name="test-model", transport=mock_transport)
        messages = [{"role": "user", "content": "Read config.json"}]
        response = client.generate_response(messages, tools=tools_schema)

        self.assertIn("tools", captured_payload)
        self.assertEqual(captured_payload["tools"], tools_schema)

        self.assertEqual(response.content, "I will read the file for you.")
        self.assertTrue(response.has_tool_calls())
        self.assertEqual(len(response.tool_calls), 1)

        tool_call = response.tool_calls[0]
        self.assertIsInstance(tool_call, ToolCall)
        self.assertEqual(tool_call.name, "read_file")
        self.assertEqual(tool_call.arguments, {"path": "config.json"})
        self.assertEqual(tool_call.id, "call_987")

    def test_multiple_tool_calls_and_dict_arguments(self):
        """Verify parsing multiple tool calls with dictionary arguments format."""
        def mock_transport(payload, headers, timeout):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": {"path": "a.txt"},
                                    },
                                },
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": '{"path": "b.txt", "content": "hello"}',
                                    },
                                },
                            ],
                        }
                    }
                ]
            }

        client = LLMClient(transport=mock_transport)
        response = client.generate_response([{"role": "user", "content": "Do work"}])

        self.assertIsNone(response.content)
        self.assertTrue(response.has_tool_calls())
        self.assertEqual(len(response.tool_calls), 2)
        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(response.tool_calls[0].arguments, {"path": "a.txt"})
        self.assertEqual(response.tool_calls[1].name, "write_file")
        self.assertEqual(response.tool_calls[1].arguments, {"path": "b.txt", "content": "hello"})

    def test_endpoint_url_formatting(self):
        """Verify endpoint URL determination logic for custom and relative URLs."""
        c1 = LLMClient(provider_url="https://api.openai.com/v1")
        self.assertEqual(c1._get_endpoint_url(), "https://api.openai.com/v1/chat/completions")

        c2 = LLMClient(provider_url="https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(c2._get_endpoint_url(), "https://openrouter.ai/api/v1/chat/completions")

        c3 = LLMClient(provider_url="default")
        self.assertEqual(c3._get_endpoint_url(), "https://openrouter.ai/api/v1/chat/completions")

    @patch("urllib.request.urlopen")
    def test_send_request_http_error(self, mock_urlopen):
        """Verify RuntimeError is raised on HTTP error from server."""
        mock_error = urllib.error.HTTPError(
            url="https://api.example.com",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=MagicMock(read=MagicMock(return_value=b'{"error": "Invalid API key"}')),
        )
        mock_urlopen.side_effect = mock_error

        client = LLMClient(provider_url="https://api.example.com")
        with self.assertRaises(RuntimeError) as ctx:
            client.generate_response([{"role": "user", "content": "test"}])

        self.assertIn("HTTP status 401", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
