"""Configuration management for the terminal coding agent."""

import os
from pathlib import Path
from typing import Optional

from coding_agent.models import AgentConfig

DEFAULT_API_KEY = ""
DEFAULT_PROVIDER_URL = "poolside/laguna-s-2.1:free"
DEFAULT_MODEL_NAME = "poolside/laguna-s-2.1:free"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_STEPS = 40

# Alias for backwards compatibility
Config = AgentConfig


def load_env_file(env_path: Optional[Path] = None) -> None:
    """Load key-value pairs from a .env file into os.environ if not already set."""
    if env_path is None:
        env_path = Path(".env")

    if not env_path.is_file():
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
    except OSError:
        pass


def load_config_from_env(env_path: Optional[Path] = None) -> AgentConfig:
    """Load AgentConfig from environment variables (or optional .env file) with sensible defaults."""
    load_env_file(env_path)

    api_key = (
        os.getenv("AGENT_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or DEFAULT_API_KEY
    )

    provider_url = (
        os.getenv("AGENT_PROVIDER_URL")
        or os.getenv("MODEL_PROVIDER_URL")
        or os.getenv("LLM_PROVIDER_URL")
        or DEFAULT_PROVIDER_URL
    )

    model_name = (
        os.getenv("AGENT_MODEL_NAME")
        or os.getenv("MODEL_NAME")
        or os.getenv("LLM_MODEL")
        or DEFAULT_MODEL_NAME
    )

    max_tokens = _parse_int(os.getenv("AGENT_MAX_TOKENS"), DEFAULT_MAX_TOKENS)
    temperature = _parse_float(os.getenv("AGENT_TEMPERATURE"), DEFAULT_TEMPERATURE)
    timeout = _parse_float(os.getenv("AGENT_TIMEOUT"), DEFAULT_TIMEOUT)
    max_steps = _parse_int(os.getenv("AGENT_MAX_STEPS"), DEFAULT_MAX_STEPS)
    system_prompt = os.getenv("AGENT_SYSTEM_PROMPT")

    return AgentConfig(
        api_key=api_key,
        provider_url=provider_url,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        max_steps=max_steps,
        system_prompt=system_prompt,
    )


# Attach from_env classmethod to AgentConfig for convenience
AgentConfig.from_env = classmethod(lambda cls, env_path=None: load_config_from_env(env_path))  # type: ignore[attr-defined]


def _parse_int(val: Optional[str], default: int) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _parse_float(val: Optional[str], default: float) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default
