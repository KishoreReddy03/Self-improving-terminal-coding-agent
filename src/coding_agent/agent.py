"""Core agent loop logic for executing reasoning and tool calls."""

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

from coding_agent.client import LLMClient
from coding_agent.memory import ExperienceMemory
from coding_agent.models import AgentConfig, AgentResult, ModelResponse
from coding_agent.registry import ToolRegistry
from coding_agent.run_record import RunOutcome, RunRecord

ApprovalCallback = Callable[[str, Dict[str, Any]], bool]


def default_terminal_approval(tool_name: str, kwargs: Dict[str, Any]) -> bool:
    """Default interactive terminal approval callback for write-capable tool execution."""
    print(f"\n[APPROVAL REQUIRED] Tool: '{tool_name}'")
    if kwargs:
        print(f"Arguments: {json.dumps(kwargs, indent=2)}")
    try:
        choice = input("Approve execution? (y/N): ").strip().lower()
        return choice in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        return False


class Agent:
    """Core Agent class managing conversation flow, model interaction, and tool execution."""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        registry: Optional[ToolRegistry] = None,
        config: Optional[AgentConfig] = None,
        auto_approve: bool = False,
        plan_mode: bool = False,
        approval_callback: Optional[ApprovalCallback] = None,
        memory: Optional[ExperienceMemory] = None,
    ) -> None:
        self.config = config or (client.config if client else AgentConfig())
        self.client = client or LLMClient(config=self.config)
        self.registry = registry or ToolRegistry.create_default()
        self.auto_approve = auto_approve
        self.plan_mode = plan_mode
        self.approval_callback = approval_callback or default_terminal_approval
        self.memory = memory

    def run(
        self,
        conversation_or_prompt: Union[List[Dict[str, Any]], str],
        max_steps: Optional[int] = None,
        plan_mode: Optional[bool] = None,
        auto_approve: Optional[bool] = None,
        approval_callback: Optional[ApprovalCallback] = None,
    ) -> AgentResult:
        """Run the core agent loop until a final response is generated or max_steps is reached."""
        if isinstance(conversation_or_prompt, str):
            task = conversation_or_prompt
            messages: List[Dict[str, Any]] = [{"role": "user", "content": task}]
        else:
            # Derive task text from the first user message for the record
            task = next(
                (m.get("content", "") for m in conversation_or_prompt if m.get("role") == "user"),
                "",
            )
            messages = [dict(m) for m in conversation_or_prompt]

        if self.config.system_prompt:
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                messages.insert(0, {"role": "system", "content": self.config.system_prompt})

        limit = max_steps if max_steps is not None else self.config.max_steps
        is_plan_mode = plan_mode if plan_mode is not None else self.plan_mode
        is_auto_approve = auto_approve if auto_approve is not None else self.auto_approve
        cb = approval_callback or self.approval_callback

        # ------------------------------------------------------------------ #
        # Initialise a RunRecord that tracks every event in this execution    #
        # ------------------------------------------------------------------ #
        record = RunRecord()
        record.record_task_start(task)

        steps = 0
        last_response_content: Optional[str] = None

        try:
            while steps < limit:
                steps += 1
                tools_schemas = self.registry.get_schemas()

                record.record_llm_call(
                    step=steps,
                    message_count=len(messages),
                    has_tools=bool(tools_schemas),
                )

                response: ModelResponse = self.client.generate_response(
                    messages=messages,
                    tools=tools_schemas if tools_schemas else None,
                )

                last_response_content = response.content

                if not response.has_tool_calls():
                    messages.append({"role": "assistant", "content": response.content})
                    record.record_final_response(response.content)
                    record.final_response = response.content
                    record.steps_taken = steps
                    record.outcome = RunOutcome.SUCCESS
                    record.messages = list(messages)
                    record.finished_at = datetime.now(timezone.utc)
                    return AgentResult(
                        final_response=response.content,
                        messages=messages,
                        steps_taken=steps,
                        completed=True,
                        run_record=record,
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
                    kwargs = tc.arguments if isinstance(tc.arguments, dict) else {}

                    if tool is None:
                        tool_output = f"Error: Tool '{tc.name}' not found."
                        # Unknown tool — no approval concept applies
                        record.record_tool_call(
                            step=steps,
                            tool_name=tc.name,
                            arguments=kwargs,
                            tool_call_id=tool_call_id,
                            approved=None,
                        )
                        record.record_tool_result(
                            step=steps,
                            tool_name=tc.name,
                            tool_call_id=tool_call_id,
                            output=tool_output,
                            is_error=True,
                        )

                    else:
                        # Policy 1: Plan Mode disables write-capable operations
                        if is_plan_mode and not tool.is_read_only:
                            tool_output = f"Error: Tool '{tool.name}' is disabled in plan mode."
                            record.record_tool_call(
                                step=steps,
                                tool_name=tool.name,
                                arguments=kwargs,
                                tool_call_id=tool_call_id,
                                approved=False,
                            )
                            record.record_tool_result(
                                step=steps,
                                tool_name=tool.name,
                                tool_call_id=tool_call_id,
                                output=tool_output,
                                is_error=True,
                            )

                        # Policy 2: Write tools require user approval unless auto_approve is set
                        elif not tool.is_read_only and not is_auto_approve:
                            approved = cb(tool.name, kwargs)
                            record.record_tool_call(
                                step=steps,
                                tool_name=tool.name,
                                arguments=kwargs,
                                tool_call_id=tool_call_id,
                                approved=approved,
                            )
                            if not approved:
                                tool_output = (
                                    f"Error: Execution of tool '{tool.name}' was denied by user."
                                )
                                record.record_tool_result(
                                    step=steps,
                                    tool_name=tool.name,
                                    tool_call_id=tool_call_id,
                                    output=tool_output,
                                    is_error=True,
                                )
                            else:
                                try:
                                    tool_output = tool.execute(**kwargs)
                                except Exception as e:
                                    tool_output = f"Error executing tool '{tool.name}': {e}"
                                record.record_tool_result(
                                    step=steps,
                                    tool_name=tool.name,
                                    tool_call_id=tool_call_id,
                                    output=(
                                        tool_output
                                        if isinstance(tool_output, str)
                                        else json.dumps(tool_output)
                                    ),
                                    is_error=str(tool_output).startswith("Error"),
                                )

                        # Policy 3: Read-only tools or auto-approved write tools execute normally
                        else:
                            # approved=None for read-only (no approval gate);
                            # approved=True for auto-approved write tools
                            approved_flag: Optional[bool] = None if tool.is_read_only else True
                            try:
                                tool_output = tool.execute(**kwargs)
                            except Exception as e:
                                tool_output = f"Error executing tool '{tool.name}': {e}"
                            record.record_tool_call(
                                step=steps,
                                tool_name=tool.name,
                                arguments=kwargs,
                                tool_call_id=tool_call_id,
                                approved=approved_flag,
                            )
                            record.record_tool_result(
                                step=steps,
                                tool_name=tool.name,
                                tool_call_id=tool_call_id,
                                output=(
                                    tool_output
                                    if isinstance(tool_output, str)
                                    else json.dumps(tool_output)
                                ),
                                is_error=str(tool_output).startswith("Error"),
                            )

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

        except Exception as exc:
            record.record_error(error_type=type(exc).__name__, message=str(exc))
            record.outcome = RunOutcome.ERROR
            record.error_message = str(exc)
            record.steps_taken = steps
            record.messages = list(messages)
            record.finished_at = datetime.now(timezone.utc)
            return AgentResult(
                final_response=last_response_content,
                messages=messages,
                steps_taken=steps,
                completed=False,
                run_record=record,
            )

        # Exhausted step budget without a final text response
        record.outcome = RunOutcome.MAX_STEPS_REACHED
        record.final_response = last_response_content
        record.steps_taken = steps
        record.messages = list(messages)
        record.finished_at = datetime.now(timezone.utc)
        return AgentResult(
            final_response=last_response_content,
            messages=messages,
            steps_taken=steps,
            completed=False,
            run_record=record,
        )
