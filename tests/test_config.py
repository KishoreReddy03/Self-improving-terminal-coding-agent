"""Tests for configuration module."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent.config import (
    Config,
    DEFAULT_API_KEY,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROVIDER_URL,
    load_env_file,
)


class TestConfig(unittest.TestCase):
    """Test Config loading and fallback logic."""

    def setUp(self):
        """Clean environment variables before each test."""
        self.env_keys = [
            "AGENT_API_KEY",
            "LLM_API_KEY",
            "OPENROUTER_API_KEY",
            "AGENT_PROVIDER_URL",
            "MODEL_PROVIDER_URL",
            "LLM_PROVIDER_URL",
            "AGENT_MODEL_NAME",
            "MODEL_NAME",
            "LLM_MODEL",
            "AGENT_MAX_TOKENS",
            "AGENT_TEMPERATURE",
            "AGENT_TIMEOUT",
        ]
        self.original_env = {k: os.environ.get(k) for k in self.env_keys}
        for k in self.env_keys:
            os.environ.pop(k, None)

    def tearDown(self):
        """Restore environment variables after each test."""
        for k, v in self.original_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    def test_default_config(self):
        """Verify defaults are loaded when environment variables are not set."""
        config = Config.from_env()
        self.assertEqual(config.api_key, DEFAULT_API_KEY)
        self.assertEqual(config.provider_url, DEFAULT_PROVIDER_URL)
        self.assertEqual(config.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(config.max_tokens, 4096)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.timeout, 60.0)

    def test_environment_variable_overrides(self):
        """Verify primary environment variables override defaults."""
        os.environ["AGENT_API_KEY"] = "custom-key-123"
        os.environ["MODEL_PROVIDER_URL"] = "https://custom-provider.api/v1"
        os.environ["MODEL_NAME"] = "custom-model-name"
        os.environ["AGENT_MAX_TOKENS"] = "2048"
        os.environ["AGENT_TEMPERATURE"] = "0.2"
        os.environ["AGENT_TIMEOUT"] = "30.5"

        config = Config.from_env()
        self.assertEqual(config.api_key, "custom-key-123")
        self.assertEqual(config.provider_url, "https://custom-provider.api/v1")
        self.assertEqual(config.model_name, "custom-model-name")
        self.assertEqual(config.max_tokens, 2048)
        self.assertEqual(config.temperature, 0.2)
        self.assertEqual(config.timeout, 30.5)

    def test_fallback_environment_variables(self):
        """Verify fallback environment variables work when primary ones are absent."""
        os.environ["OPENROUTER_API_KEY"] = "openrouter-key-abc"
        os.environ["LLM_PROVIDER_URL"] = "https://openrouter.ai/api/v1"
        os.environ["LLM_MODEL"] = "fallback-model"

        config = Config.from_env()
        self.assertEqual(config.api_key, "openrouter-key-abc")
        self.assertEqual(config.provider_url, "https://openrouter.ai/api/v1")
        self.assertEqual(config.model_name, "fallback-model")

    def test_invalid_type_conversion_uses_defaults(self):
        """Verify invalid int/float strings fall back to default values safely."""
        os.environ["AGENT_MAX_TOKENS"] = "not-an-int"
        os.environ["AGENT_TEMPERATURE"] = "not-a-float"
        os.environ["AGENT_TIMEOUT"] = "invalid"

        config = Config.from_env()
        self.assertEqual(config.max_tokens, 4096)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.timeout, 60.0)

    def test_load_env_file(self):
        """Verify loading configuration from a .env file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_file = Path(tmp_dir) / ".env"
            env_file.write_text(
                "AGENT_API_KEY=file-api-key\nMODEL_NAME=file-model-name\n",
                encoding="utf-8",
            )
            config = Config.from_env(env_path=env_file)
            self.assertEqual(config.api_key, "file-api-key")
            self.assertEqual(config.model_name, "file-model-name")


if __name__ == "__main__":
    unittest.main()
