"""Tool call parser and stream accumulator for converting LLM provider responses into internal representations."""

import json
from typing import Any, Dict, List, Optional, Union

from coding_agent.models import ModelResponse, ToolCall


class ResponseParser:
    """Parser for converting provider LLM responses into internal ModelResponse & ToolCall objects."""

    @staticmethod
    def parse(raw_response: Dict[str, Any]) -> ModelResponse:
        """Parse a complete provider API response dictionary into a ModelResponse."""
        if not raw_response or not isinstance(raw_response, dict):
            return ModelResponse(content=None, tool_calls=[], raw_response=raw_response)

        choices = raw_response.get("choices", [])
        if not choices:
            return ModelResponse(content=None, tool_calls=[], raw_response=raw_response)

        first_choice = choices[0]
        msg_obj = first_choice.get("message") or first_choice.get("delta") or {}
        content = msg_obj.get("content")

        parsed_tool_calls: List[ToolCall] = []
        raw_tool_calls = msg_obj.get("tool_calls", [])
        for tc in raw_tool_calls:
            tc_id = tc.get("id")
            func_data = tc.get("function", {})
            name = func_data.get("name", "")
            args_raw = func_data.get("arguments", {})

            args = ResponseParser._parse_arguments(args_raw)
            parsed_tool_calls.append(ToolCall(name=name, arguments=args, id=tc_id))

        return ModelResponse(
            content=content,
            tool_calls=parsed_tool_calls,
            raw_response=raw_response,
        )

    @staticmethod
    def _parse_arguments(args_raw: Any) -> Dict[str, Any]:
        """Parse raw tool arguments (string or dict) into a dictionary."""
        if isinstance(args_raw, str):
            args_str = args_raw.strip()
            if not args_str:
                return {}
            try:
                parsed = json.loads(args_str)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                return {"raw_arguments": args_raw}
        elif isinstance(args_raw, dict):
            return args_raw
        elif args_raw is None:
            return {}
        else:
            return {"value": args_raw}


class StreamToolCallAccumulator:
    """Accumulator for processing streaming response chunks and assembling tool calls & text content."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the accumulator state for fresh stream processing."""
        self._content_parts: List[str] = []
        self._tool_call_buffers: Dict[int, Dict[str, Any]] = {}

    def add_chunk(self, chunk: Union[Dict[str, Any], str]) -> None:
        """Add a streaming chunk dictionary or JSON string to the accumulator."""
        if isinstance(chunk, str):
            chunk_str = chunk.strip()
            if not chunk_str:
                return
            try:
                chunk_dict = json.loads(chunk_str)
            except json.JSONDecodeError:
                self._content_parts.append(chunk)
                return
        elif isinstance(chunk, dict):
            chunk_dict = chunk
        else:
            return

        choices = chunk_dict.get("choices", [])
        if not choices:
            return

        first_choice = choices[0]
        delta = first_choice.get("delta") or first_choice.get("message") or {}

        content_fragment = delta.get("content")
        if content_fragment:
            self._content_parts.append(content_fragment)

        tool_calls_delta = delta.get("tool_calls", [])
        for idx, tc_delta in enumerate(tool_calls_delta):
            tc_index = tc_delta.get("index")
            if tc_index is None:
                tc_index = idx

            if tc_index not in self._tool_call_buffers:
                self._tool_call_buffers[tc_index] = {
                    "id": None,
                    "name_parts": [],
                    "arguments_parts": [],
                }

            buffer = self._tool_call_buffers[tc_index]

            if tc_delta.get("id"):
                buffer["id"] = tc_delta["id"]

            func_delta = tc_delta.get("function", {})
            if "name" in func_delta and func_delta["name"]:
                buffer["name_parts"].append(func_delta["name"])

            if "arguments" in func_delta and func_delta["arguments"]:
                args_part = func_delta["arguments"]
                if isinstance(args_part, str):
                    buffer["arguments_parts"].append(args_part)
                elif isinstance(args_part, dict):
                    buffer["arguments_parts"].append(json.dumps(args_part))

    def to_model_response(self) -> ModelResponse:
        """Assemble accumulated content and tool calls into a finalized ModelResponse."""
        content = "".join(self._content_parts) if self._content_parts else None

        parsed_tool_calls: List[ToolCall] = []
        for index in sorted(self._tool_call_buffers.keys()):
            buffer = self._tool_call_buffers[index]
            tc_id = buffer["id"]
            name = "".join(buffer["name_parts"])
            raw_args_str = "".join(buffer["arguments_parts"])

            args = ResponseParser._parse_arguments(raw_args_str)
            parsed_tool_calls.append(ToolCall(name=name, arguments=args, id=tc_id))

        return ModelResponse(
            content=content,
            tool_calls=parsed_tool_calls,
        )

