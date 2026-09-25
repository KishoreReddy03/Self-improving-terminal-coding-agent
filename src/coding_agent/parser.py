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
