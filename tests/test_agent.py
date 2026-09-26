"""Tests for core agent loop logic."""

import unittest
from typing import Any, Dict, List, Optional

from coding_agent.agent import Agent
from coding_agent.client import LLMClient
from coding_agent.models import AgentConfig, AgentResult, ModelResponse, ToolCall
from coding_agent.registry import ToolRegistry
from coding_agent.tools import Tool


class FakeTool(Tool):
    """Fake tool implementation for testing agent loop tool execution."""

    def __init__(self, name: str, return_value: Any = "success") -> None:
        super().__init__(
            name=name,
            description=f"Fake tool {name}",
            parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
        )
        self.return_value = return_value
        self.calls: List[Dict[str, Any]] = []

    def execute(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.return_value, Exception):
            raise self.return_value
        return self.return_value


class FakeLLMClient(LLMClient):
    """Fake LLM client returning a predefined sequence of ModelResponse objects."""

    def __init__(self, responses: List[ModelResponse], config: Optional[AgentConfig] = None) -> None:
        super().__init__(config=config or AgentConfig())
        self.responses = list(responses)
        self.call_count = 0
        self.received_messages_history: List[List[Dict[str, Any]]] = []

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        self.received_messages_history.append([dict(m) for m in messages])
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return ModelResponse(content="Default fallback final response", tool_calls=[])


class TestAgentLoop(unittest.TestCase):
    """Test Agent core loop execution, tool dispatching, and termination conditions."""

    def test_single_step_text_response(self):
        """Verify agent completes in 1 step when model returns a normal text response."""
        client = FakeLLMClient([ModelResponse(content="Hello! How can I help?")])
        registry = ToolRegistry()
        agent = Agent(client=client, registry=registry)

        result = agent.run("Hi")

        self.assertIsInstance(result, AgentResult)
        self.assertTrue(result.completed)
        self.assertEqual(result.steps_taken, 1)
        self.assertEqual(result.final_response, "Hello! How can I help?")
        self.assertEqual(len(result.messages), 2)
        self.assertEqual(result.messages[0], {"role": "user", "content": "Hi"})
        self.assertEqual(result.messages[1], {"role": "assistant", "content": "Hello! How can I help?"})

    def test_tool_call_execution_and_loop_continuation(self):
        """Verify agent executes requested tool, appends output to conversation, and receives final answer."""
        fake_tool = FakeTool(name="read_file", return_value="file content line 1")
        registry = ToolRegistry()
        registry.register(fake_tool)

        responses = [
            ModelResponse(
                content="I will read the file.",
                tool_calls=[ToolCall(name="read_file", arguments={"path": "foo.txt"}, id="call_1")],
            ),
            ModelResponse(content="The file contains: file content line 1"),
        ]
        client = FakeLLMClient(responses)
        agent = Agent(client=client, registry=registry)

        result = agent.run("Read foo.txt")

        self.assertTrue(result.completed)
        self.assertEqual(result.steps_taken, 2)
        self.assertEqual(result.final_response, "The file contains: file content line 1")
        self.assertEqual(len(fake_tool.calls), 1)
        self.assertEqual(fake_tool.calls[0], {"path": "foo.txt"})

        # Check message history structure: user -> assistant (tool_call) -> tool (result) -> assistant (final)
        messages = result.messages
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["tool_call_id"], "call_1")
        self.assertEqual(messages[2]["content"], "file content line 1")
        self.assertEqual(messages[3]["role"], "assistant")

    def test_multiple_tool_calls_in_single_step(self):
        """Verify agent handles multiple tool calls requested in a single assistant response."""
        t1 = FakeTool(name="tool_a", return_value="res_a")
        t2 = FakeTool(name="tool_b", return_value="res_b")
        registry = ToolRegistry()
        registry.register(t1)
        registry.register(t2)

        responses = [
            ModelResponse(
                content="Running both tools.",
                tool_calls=[
                    ToolCall(name="tool_a", arguments={"arg": "1"}, id="call_a"),
                    ToolCall(name="tool_b", arguments={"arg": "2"}, id="call_b"),
                ],
            ),
            ModelResponse(content="Both tools completed successfully."),
        ]
        client = FakeLLMClient(responses)
        agent = Agent(client=client, registry=registry)

        result = agent.run("Run both")

        self.assertTrue(result.completed)
        self.assertEqual(result.steps_taken, 2)
        self.assertEqual(len(t1.calls), 1)
        self.assertEqual(len(t2.calls), 1)

        tool_messages = [m for m in result.messages if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 2)
        self.assertEqual(tool_messages[0]["content"], "res_a")
        self.assertEqual(tool_messages[1]["content"], "res_b")

    def test_unknown_tool_call_error_handling(self):
        """Verify requesting an unregistered tool appends an error message to conversation."""
        registry = ToolRegistry()  # Empty registry
        responses = [
            ModelResponse(
                content="Trying unknown tool.",
                tool_calls=[ToolCall(name="unknown_tool", arguments={}, id="call_missing")],
            ),
            ModelResponse(content="Handled missing tool error."),
        ]
        client = FakeLLMClient(responses)
        agent = Agent(client=client, registry=registry)

        result = agent.run("Do action")

        self.assertTrue(result.completed)
        tool_msg = [m for m in result.messages if m["role"] == "tool"][0]
        self.assertIn("Error: Tool 'unknown_tool' not found", tool_msg["content"])

    def test_tool_execution_exception_handling(self):
        """Verify tool execution exceptions are caught and reported as error messages."""
        failing_tool = FakeTool(name="fail_tool", return_value=RuntimeError("Disk read error"))
        registry = ToolRegistry()
        registry.register(failing_tool)

        responses = [
            ModelResponse(
                content="Executing failing tool.",
                tool_calls=[ToolCall(name="fail_tool", arguments={}, id="call_fail")],
            ),
            ModelResponse(content="Handled exception."),
        ]
        client = FakeLLMClient(responses)
        agent = Agent(client=client, registry=registry)

        result = agent.run("Try fail")

        tool_msg = [m for m in result.messages if m["role"] == "tool"][0]
        self.assertIn("Error executing tool 'fail_tool': Disk read error", tool_msg["content"])

    def test_max_steps_iteration_limit(self):
        """Verify loop terminates with completed=False when reaching max_steps limit."""
        repeating_response = ModelResponse(
            content="Infinite tool call",
            tool_calls=[ToolCall(name="dummy_tool", arguments={})],
        )
        client = FakeLLMClient([repeating_response] * 10)
        registry = ToolRegistry()
        registry.register(FakeTool(name="dummy_tool"))

        config = AgentConfig(max_steps=3)
        agent = Agent(client=client, registry=registry, config=config)

        result = agent.run("Loop test")

        self.assertFalse(result.completed)
        self.assertEqual(result.steps_taken, 3)

    def test_system_prompt_prepending(self):
        """Verify system prompt is prepended to conversation if configured."""
        client = FakeLLMClient([ModelResponse(content="System prompt acknowledged.")])
        config = AgentConfig(system_prompt="You are a senior python developer.")
        agent = Agent(client=client, config=config, registry=ToolRegistry())

        result = agent.run("Hello")

        first_msg = result.messages[0]
        self.assertEqual(first_msg["role"], "system")
        self.assertEqual(first_msg["content"], "You are a senior python developer.")


if __name__ == "__main__":
    unittest.main()
