"""LLM client layer for isolating provider interaction details and supporting tool calls."""

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from coding_agent.models import AgentConfig, ModelResponse, ToolCall
from coding_agent.parser import ResponseParser


class LLMClient:
    """Client for interacting with LLM model providers while isolating API and auth details."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        api_key: Optional[str] = None,
        provider_url: Optional[str] = None,
        model_name: Optional[str] = None,
        transport: Optional[Callable[[Dict[str, Any], Dict[str, str], float], Dict[str, Any]]] = None,
    ) -> None:
        if config is None:
            config = AgentConfig()

        self.config = config
        self.api_key = api_key if api_key is not None else config.api_key
        self.provider_url = provider_url if provider_url is not None else config.provider_url
        self.model_name = model_name if model_name is not None else config.model_name
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.timeout = config.timeout
        self._transport = transport

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        """Send chat messages and optional tool definitions to the LLM and return ModelResponse."""
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }

        if tools:
            payload["tools"] = tools

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self._transport is not None:
            raw_response = self._transport(payload, headers, self.timeout)
        else:
            raw_response = self._send_request(payload, headers)

        return self._parse_response(raw_response)

    def _get_endpoint_url(self) -> str:
        """Determine full API endpoint URL from provider_url configuration."""
        url = self.provider_url.strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            url = "https://openrouter.ai/api/v1/chat/completions"
        elif not url.endswith("/chat/completions"):
            url = url.rstrip("/") + "/chat/completions"
        return url

    def _send_request(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Perform HTTP POST request to provider endpoint using urllib."""
        endpoint = self._get_endpoint_url()
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_data = resp.read().decode("utf-8")
                return json.loads(resp_data)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API request failed with HTTP status {e.code}: {err_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"LLM API network error: {e.reason}") from e

    def _parse_response(self, raw_response: Dict[str, Any]) -> ModelResponse:
        """Parse raw API JSON response dictionary into a structured ModelResponse."""
        return ResponseParser.parse(raw_response)
