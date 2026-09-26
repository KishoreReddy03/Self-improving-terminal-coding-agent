"""Core agent loop logic for executing reasoning and tool calls."""

import json
from typing import Any, Dict, List, Optional, Union

from coding_agent.client import LLMClient
from coding_agent.models import AgentConfig, AgentResult, ModelResponse
from coding_agent.registry import ToolRegistry


class Agent:
    """Core Agent class managing conversation flow, model interaction, and tool execution."""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        registry: Optional[ToolRegistry] = None,
        config: Optional[AgentConfig] = None,
    ) -> None:
        self.config = config or (client.config if client else AgentConfig())
        self.client = client or LLMClient(config=self.config)
        self.registry = registry or ToolRegistry.create_default()

    def run(
        self,
        conversation_or_prompt: Union[List[Dict[str, Any]], str],
        max_steps: Optional[int] = None,
    ) -> AgentResult:
        """Run the core agent loop until a final response is generated or max_steps is reached."""
        if isinstance(conversation_or_prompt, str):
            messages: List[Dict[str, Any]] = [{"role": "user", "content": conversation_or_prompt}]
        else:
            messages = [dict(m) for m in conversation_or_prompt]

        if self.config.system_prompt:
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                messages.insert(0, {"role": "system", "content": self.config.system_prompt})

        limit = max_steps if max_steps is not None else self.config.max_steps
        steps = 0
        last_response_content: Optional[str] = None

        while steps < limit:
            steps += 1
            tools_schemas = self.registry.get_schemas()

            response: ModelResponse = self.client.generate_response(
                messages=messages,
                tools=tools_schemas if tools_schemas else None,
            )

            last_response_content = response.content

            if not response.has_tool_calls():
                messages.append({"role": "assistant", "content": response.content})
                return AgentResult(
                    final_response=response.content,
                    messages=messages,
                    steps_taken=steps,
                    completed=True,
                )

            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc.id or f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                            if isinstance(tc.arguments, dict)
                            else str(tc.arguments),
                        },
                    }
                    for i, tc in enumerate(response.tool_calls)
                ],
            }
            messages.append(assistant_msg)

            for i, tc in enumerate(response.tool_calls):
                tool_call_id = tc.id or f"call_{i}"
                tool = self.registry.get(tc.name)

                if tool is None:
                    tool_output = f"Error: Tool '{tc.name}' not found."
                else:
                    kwargs = tc.arguments if isinstance(tc.arguments, dict) else {}
                    try:
                        tool_output = tool.execute(**kwargs)
                    except Exception as e:
                        tool_output = f"Error executing tool '{tc.name}': {e}"

                if not isinstance(tool_output, str):
                    tool_output = json.dumps(tool_output)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tc.name,
                        "content": tool_output,
                    }
                )

        return AgentResult(
            final_response=last_response_content,
            messages=messages,
            steps_taken=steps,
            completed=False,
        )
