"""CLI entry point for the terminal coding agent."""

import sys
from coding_agent.config import Config


def main() -> int:
    """Main CLI execution entry point."""
    config = Config.from_env()
    print("Terminal Coding Agent initialized successfully.")
    print(f"Model Provider URL: {config.provider_url}")
    print(f"Model Name:         {config.model_name}")
    masked_key = (
        config.api_key[:8] + "..." + config.api_key[-4:]
        if len(config.api_key) > 12
        else "***"
    )
    print(f"API Key:            {masked_key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
