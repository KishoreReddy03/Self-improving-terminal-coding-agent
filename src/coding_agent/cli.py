"""CLI entry point for the terminal coding agent."""

import argparse
import sys
from typing import Any, Dict, List, Optional

from coding_agent.agent import Agent
from coding_agent.config import load_config_from_env


PLAN_MODE_DIRECTIVE = (
    "\n\n[PLAN MODE INSTRUCTION]: Please first analyze the request and outline a clear, "
    "step-by-step implementation plan before proceeding with execution."
)


def run_interactive_session(agent: Agent, start_in_plan_mode: bool = False) -> int:
    """Run interactive REPL terminal loop for the agent."""
    conversation: List[Dict[str, Any]] = []

    print("Terminal Coding Agent v0.1.0")
    print("Type /plan to toggle plan mode, /clear to reset history, or /exit to quit.\n")

    while True:
        try:
            user_input = input("agent> ").strip()
        except EOFError:
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print("Goodbye!")
            break

        conversation.append({"role": "user", "content": user_input})

        try:
            result = agent.run(conversation)
            conversation = result.messages
            if result.final_response:
                print(f"\n{result.final_response}\n")
        except Exception as e:
            print(f"\nError during agent execution: {e}\n")

    return 0


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI execution entry point."""
    config = load_config_from_env()
    agent = Agent(config=config)
    return run_interactive_session(agent)


if __name__ == "__main__":
    sys.exit(main())
